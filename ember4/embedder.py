"""Model-aware ONNX text embedder for ember-4.

Supports multiple embedding models with task-prefix discipline. Each
ModelSpec carries its own dimension, max length, ONNX file, input tensor
names, and per-role prefix strings. Embedders are constructed against a
spec; the prefix is applied transparently when you call embed_query /
embed_document / embed_cluster.

Currently registered:
- MINILM_L6_SPEC   — sentence-transformers/all-MiniLM-L6-v2, 384-dim, no prefixes
- NOMIC_V1_5_SPEC  — nomic-ai/nomic-embed-text-v1.5, 768-dim, four task prefixes

Why prefixes matter: Nomic was contrastively trained with explicit task
instructions. A query embedded as `search_query: foo` lives in a different
region of the manifold than the same string embedded as `clustering: foo`.
Conflating them is the canonical Nomic footgun.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ember4.config import get_data_dir

logger = logging.getLogger(__name__)


# ─── Model specifications ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    """Static description of an embedding model.

    name: short identifier used in DB / config.
    hf_repo: HuggingFace repo id for download.
    onnx_filename: relative path to the ONNX file in the repo.
    tokenizer_filename: relative path to tokenizer.json.
    dimension: output vector dimensionality.
    max_length: tokenizer truncation length (model can usually handle more).
    input_names: tuple of ONNX input tensor names the model expects.
    pooling: 'mean' | 'cls'.
    prefixes: dict mapping role → prefix string (must include trailing space
              if Nomic-style). Roles not present default to empty string.
    """
    name: str
    hf_repo: str
    onnx_filename: str
    tokenizer_filename: str
    dimension: int
    max_length: int
    input_names: tuple[str, ...]
    pooling: str = "mean"
    prefixes: dict[str, str] = field(default_factory=dict)

    def prefix_for(self, role: str) -> str:
        return self.prefixes.get(role, "")

    def model_dir(self) -> Path:
        return get_data_dir() / "models" / self.name


# Ember-3's legacy model — kept around for A/B comparison and back-compat
MINILM_L6_SPEC = ModelSpec(
    name="all-MiniLM-L6-v2",
    hf_repo="sentence-transformers/all-MiniLM-L6-v2",
    onnx_filename="onnx/model.onnx",
    tokenizer_filename="tokenizer.json",
    dimension=384,
    max_length=256,
    input_names=("input_ids", "attention_mask", "token_type_ids"),
    pooling="mean",
    prefixes={},  # MiniLM has no prefix discipline; all roles use empty prefix
)

# Ember-4's primary embedder — Nomic with task-prefix discipline.
# Using int8 quantized so it runs comfortably on laptops.
NOMIC_V1_5_SPEC = ModelSpec(
    name="nomic-embed-text-v1.5",
    hf_repo="nomic-ai/nomic-embed-text-v1.5",
    onnx_filename="onnx/model_int8.onnx",
    tokenizer_filename="tokenizer.json",
    dimension=768,
    max_length=2048,  # well under Nomic's native 8192; bounds memory on laptop
    # The ONNX export retains token_type_ids in the signature even though the
    # model uses rotary position embeddings — we feed zeros for it.
    input_names=("input_ids", "attention_mask", "token_type_ids"),
    pooling="mean",
    prefixes={
        "query": "search_query: ",
        "document": "search_document: ",
        "cluster": "clustering: ",
        "classify": "classification: ",
    },
)


# ─── Embedder ─────────────────────────────────────────────────────────────────

class Embedder:
    """ONNX text embedder bound to a single ModelSpec.

    Construct one Embedder per model. The spec controls dimension, prefixes,
    pooling, and input tensor names; the Embedder itself only knows how to
    download, tokenize, run inference, and L2-normalize.
    """

    def __init__(self, spec: ModelSpec = NOMIC_V1_5_SPEC):
        self.spec = spec
        self.model_dir = spec.model_dir()
        self._session = None
        self._tokenizer = None

    # ─── Lazy model loading ───────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """Download model files from HuggingFace if not cached locally."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / Path(self.spec.onnx_filename).name
        tokenizer_path = self.model_dir / "tokenizer.json"

        if model_path.exists() and tokenizer_path.exists():
            return

        from huggingface_hub import hf_hub_download

        if not model_path.exists():
            logger.info(f"downloading {self.spec.hf_repo}/{self.spec.onnx_filename}")
            downloaded = hf_hub_download(
                repo_id=self.spec.hf_repo,
                filename=self.spec.onnx_filename,
                local_dir=self.model_dir,
            )
            p = Path(downloaded)
            if p != model_path:
                p.rename(model_path)

        if not tokenizer_path.exists():
            logger.info(f"downloading {self.spec.hf_repo}/{self.spec.tokenizer_filename}")
            downloaded = hf_hub_download(
                repo_id=self.spec.hf_repo,
                filename=self.spec.tokenizer_filename,
                local_dir=self.model_dir,
            )
            p = Path(downloaded)
            if p != tokenizer_path:
                p.rename(tokenizer_path)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._ensure_model()
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            tok.enable_truncation(max_length=self.spec.max_length)
            # Pad to the longest in the batch (no fixed length). Padding to
            # spec.max_length would force every sequence to 2048 tokens for
            # Nomic — wasteful when most embers are <500 tokens. Multiple-of-8
            # alignment is friendly to AVX/Apple SIMD without over-padding.
            tok.enable_padding(pad_id=0, pad_token="[PAD]", pad_to_multiple_of=8)
            self._tokenizer = tok
        return self._tokenizer

    @property
    def session(self):
        if self._session is None:
            self._ensure_model()
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.enable_cpu_mem_arena = False
            model_path = self.model_dir / Path(self.spec.onnx_filename).name
            self._session = ort.InferenceSession(
                str(model_path), sess_options=opts,
            )
            # Verify the model exposes the inputs we expect
            got = {i.name for i in self._session.get_inputs()}
            expected = set(self.spec.input_names)
            missing = expected - got
            if missing:
                raise RuntimeError(
                    f"{self.spec.name}: ONNX model missing expected inputs "
                    f"{missing}; got {sorted(got)}"
                )
        return self._session

    # ─── Core inference ───────────────────────────────────────────────────

    def _run_onnx(self, texts: list[str]) -> np.ndarray:
        """Run inference. Returns (N, dim) float32, L2-normalized."""
        # encode_batch pads to the longest text in this batch, producing a
        # rectangular tensor. The list-comp over .encode() pads each call
        # independently and yields ragged shapes — silent footgun.
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array(
            [e.attention_mask for e in encoded], dtype=np.int64
        )

        feed: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self.spec.input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        output = self.session.run(None, feed)
        last_hidden = output[0]  # (N, seq_len, dim)

        if self.spec.pooling == "mean":
            mask_expanded = np.broadcast_to(
                np.expand_dims(attention_mask, -1), last_hidden.shape
            )
            embeddings = np.sum(last_hidden * mask_expanded, axis=1) / np.clip(
                mask_expanded.sum(axis=1), a_min=1e-9, a_max=None
            )
        elif self.spec.pooling == "cls":
            embeddings = last_hidden[:, 0, :]
        else:
            raise ValueError(f"unknown pooling: {self.spec.pooling}")

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return (embeddings / norms).astype(np.float32)

    # ─── Public API: role-aware embedding ─────────────────────────────────

    def embed(self, text: str, role: str = "document") -> np.ndarray:
        """Encode one text. role ∈ {'query', 'document', 'cluster', 'classify'}.

        Default role='document' matches the most common use (storing embers).
        For MiniLM (no prefixes) the role has no effect.
        """
        prefix = self.spec.prefix_for(role)
        return self._run_onnx([prefix + text])[0]

    def batch_embed(self, texts: list[str], role: str = "document") -> np.ndarray:
        prefix = self.spec.prefix_for(role)
        return self._run_onnx([prefix + t for t in texts])

    # Convenience aliases — explicit about intent at the call site
    def embed_query(self, text: str) -> np.ndarray:
        return self.embed(text, role="query")

    def embed_document(self, text: str) -> np.ndarray:
        return self.embed(text, role="document")

    def embed_cluster(self, text: str) -> np.ndarray:
        return self.embed(text, role="cluster")

    def batch_embed_query(self, texts: list[str]) -> np.ndarray:
        return self.batch_embed(texts, role="query")

    def batch_embed_document(self, texts: list[str]) -> np.ndarray:
        return self.batch_embed(texts, role="document")

    def batch_embed_cluster(self, texts: list[str]) -> np.ndarray:
        return self.batch_embed(texts, role="cluster")


def serialize_vector(vec: np.ndarray) -> bytes:
    """Serialize a float32 vector to bytes for sqlite-vec."""
    return vec.astype(np.float32).flatten().tobytes()
