#!/usr/bin/env bash
# Start one split in tmux; --dry-run prints the command without starting anything.
set -euo pipefail
usage() {
  echo "Usage: $0 <s0|s1|s2> <physical_gpu_id> [--dry-run]"
}
if [[ "${1:-}" == "--help" ]]; then usage; exit 0; fi
if (( $# < 2 || $# > 3 )); then usage >&2; exit 2; fi
SPLIT="$1"
GPU_ID="$2"
[[ "$SPLIT" =~ ^s[012]$ && "$GPU_ID" =~ ^[0-9]+$ ]] || { usage >&2; exit 2; }
[[ $# == 2 || "$3" == "--dry-run" ]] || { usage >&2; exit 2; }
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="kg-pharm-${SPLIT}-seed42"
printf -v RUN_COMMAND 'bash %q %q %q' "$REPO_DIR/tools/run_kg_pharm_pipeline.sh" "$SPLIT" "$GPU_ID"
if [[ "${3:-}" == "--dry-run" ]]; then
  printf 'Session: %s
Working directory: %s
Command: %s
' "$SESSION" "$REPO_DIR" "$RUN_COMMAND"
  exit 0
fi
command -v tmux >/dev/null || { echo "tmux is not installed" >&2; exit 1; }
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists; attach with: tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -c "$REPO_DIR" "$RUN_COMMAND"
printf 'Started: %s
Attach: tmux attach -t %s
Log: %s/work_dirs/new_%s_kg_gnn_pharmacophore_n1/pipeline.log
' "$SESSION" "$SESSION" "$REPO_DIR" "$SPLIT"
