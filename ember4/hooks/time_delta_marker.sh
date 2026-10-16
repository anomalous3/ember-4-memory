#!/bin/bash
# UserPromptSubmit hook: emit a minimal time-delta marker.
#
# Reads the timestamp written by stop_heartbeat.sh when the previous
# turn ended. Computes wall-clock delta and outputs a compact marker
# prepended to the user's prompt (via stdout; Claude Code injects
# hook stdout as additional context).
#
# Output format (when delta is available):
#   [time: 14:23 PDT · +47m since last turn]
# Local time + timezone abbreviation; %Z renders the TZ name from the
# system locale (PDT/PST/CDT/etc).
#
# Design constraints (discussed in session 2026-04-22):
#  - No call-to-action framing ("anything to report?" etc.)
#  - No summarization of state (no log tails, GPU stats, etc.)
#  - Silent on first turn of session (no last_stop_time yet)
#  - Silent if delta is <1m (ambient within-turn work)
#
# Purpose: anchor the model's sense of time passing in marathon sessions
# where sequential events don't convey duration.

LAST_STOP_FILE="$HOME/.claude/last_stop_time"

if [ ! -f "$LAST_STOP_FILE" ]; then
    exit 0  # First turn, no prior stop time
fi

last=$(cat "$LAST_STOP_FILE" 2>/dev/null)
if [ -z "$last" ] || ! [ "$last" -gt 0 ] 2>/dev/null; then
    exit 0  # Corrupt or empty
fi

now=$(date +%s)
delta=$(( (now - last) ))
delta_min=$(( delta / 60 ))

# Suppress for sub-minute deltas (within-turn noise)
if [ "$delta_min" -lt 1 ]; then
    exit 0
fi

# Format the anchor. Compact, factual, no imperative. Local time + TZ.
time_str=$(date +'%H:%M %Z')

if [ "$delta_min" -lt 60 ]; then
    echo "[time: ${time_str} · +${delta_min}m since last turn]"
elif [ "$delta_min" -lt 1440 ]; then
    hours=$(( delta_min / 60 ))
    mins=$(( delta_min % 60 ))
    echo "[time: ${time_str} · +${hours}h${mins}m since last turn]"
else
    days=$(( delta_min / 1440 ))
    echo "[time: ${time_str} · +${days}d since last turn]"
fi

exit 0
