#!/bin/bash
# Stop hook: write the wall-clock time this turn ended. time_delta_marker.sh
# (UserPromptSubmit) reads it and surfaces "[time: HH:MM · +Nm since last turn]"
# on the next turn, giving the model a sense of time passing while it was off.
echo "$(date +%s)" > "$HOME/.claude/last_stop_time" 2>/dev/null || true
exit 0
