#!/usr/bin/env bash
set -euo pipefail
SPLIT="${1:-s0}"
GPU_ID="${2:-1}"
[[ "$SPLIT" =~ ^s[012]$ ]] || { echo "Usage: $0 [s0|s1|s2] [GPU]" >&2; exit 2; }
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
PYTHON="${ZERODDI_PYTHON:-/home/wumengying/miniconda3/envs/zeroddi/bin/python}"
CONFIG="configs/new_${SPLIT}_kg_gnn_pharmacophore_n1.py"
WORK_DIR="work_dirs/new_${SPLIT}_kg_gnn_pharmacophore_n1"
mkdir -p "$WORK_DIR"
exec 9>"$WORK_DIR/.pipeline.lock"
flock -n 9 || { echo "Pipeline already running" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec > >(tee -a "$WORK_DIR/pipeline.log") 2>&1
CHECKPOINT="$WORK_DIR/model_parameter/model_best_epoch100_seen42.pkl"
if [[ ! -f "$WORK_DIR/.train_complete" ]]; then
  "$PYTHON" main.py --config "$CONFIG" --device cuda:0 --seednumber 42
  test -f "$CHECKPOINT"
  touch "$WORK_DIR/.train_complete"
fi
if [[ ! -f "$WORK_DIR/.eval_complete" ]]; then
  "$PYTHON" main.py --config "$CONFIG" --device cuda:0 --seednumber 42 --seen_para "$CHECKPOINT"
  touch "$WORK_DIR/.eval_complete"
fi
