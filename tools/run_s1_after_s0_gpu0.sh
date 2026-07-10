#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
S0_PATTERN="${S0_PATTERN:-main.py.*configs/new_s0_reverse_kg.py}"
S1_CONFIG="${S1_CONFIG:-configs/new_s1_reverse_kg.py}"
S1_STDOUT="${S1_STDOUT:-work_dirs/new_s1_reverse_kg_stdout.log}"

usage() {
  cat <<'EOF'
Usage:
  tools/run_s1_after_s0_gpu0.sh [--gpu 0] [--poll-seconds 60] [--s0-pattern REGEX] [--] [extra main.py args...]

Environment overrides:
  GPU_ID          GPU index to use for S1. Default: 0
  POLL_SECONDS    Seconds between S0 checks. Default: 60
  S0_PATTERN      pgrep -f regex used to detect the running S0 job.
                  Default: main.py.*configs/new_s0_reverse_kg.py
  S1_CONFIG       Config to run after S0 finishes. Default: configs/new_s1_reverse_kg.py
  S1_STDOUT       S1 stdout/stderr log. Default: work_dirs/new_s1_reverse_kg_stdout.log

Examples:
  nohup tools/run_s1_after_s0_gpu0.sh > work_dirs/s1_after_s0_watcher.log 2>&1 &
  GPU_ID=1 tools/run_s1_after_s0_gpu0.sh -- --seednumber 42
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --s0-pattern)
      S0_PATTERN="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

mkdir -p "$(dirname "$S1_STDOUT")"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] Watching for S0 process pattern: ${S0_PATTERN}"
echo "[$(timestamp)] S1 will run on cuda:${GPU_ID} with config ${S1_CONFIG}"

while pgrep -u "$USER" -f "$S0_PATTERN" >/dev/null; do
  pids="$(pgrep -u "$USER" -f "$S0_PATTERN" | tr '\n' ' ')"
  echo "[$(timestamp)] S0 still running. PID(s): ${pids} Sleeping ${POLL_SECONDS}s."
  sleep "$POLL_SECONDS"
done

if pgrep -u "$USER" -f "main.py.*${S1_CONFIG}" >/dev/null; then
  echo "[$(timestamp)] S1 is already running; not starting another copy."
  exit 0
fi

echo "[$(timestamp)] S0 finished. Starting S1..."
echo "[$(timestamp)] Command: python main.py --config ${S1_CONFIG} --device cuda:${GPU_ID} $*" | tee -a "$S1_STDOUT"

set +e
python main.py --config "$S1_CONFIG" --device "cuda:${GPU_ID}" "$@" >> "$S1_STDOUT" 2>&1
status=$?
set -e

echo "[$(timestamp)] S1 command exited with status ${status}. Log: ${S1_STDOUT}"
exit "$status"
