#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 <s0|s1|s2> [physical_gpu_id]" >&2
  exit 2
fi

SPLIT="$1"
GPU_ID="${2:-4}"
if [[ ! "$SPLIT" =~ ^s[012]$ ]]; then
  echo "Unsupported split: $SPLIT" >&2
  exit 2
fi

REPO_DIR="/home/wumengying/ZeroDDI"
COMMON_KG_DIR="$REPO_DIR/data/KnowDDI/drugbank_true_s0/kg"
KG_FILE="$COMMON_KG_DIR/molecbionet_pair_graph_s0.sqlite"
WORK_DIR="$REPO_DIR/work_dirs/new_${SPLIT}_reverse_kg_gnn"
CONFIG="configs/new_${SPLIT}_reverse_kg_gnn.py"
CHECKPOINT="$WORK_DIR/model_parameter/seen_model_best_epoch100_seed42.pkl"
PIPELINE_LOG="$WORK_DIR/pipeline.log"
TRAIN_MARKER="$WORK_DIR/.train_complete_seed42"
EVAL_MARKER="$WORK_DIR/.test_eval_complete_seed42"

mkdir -p "$WORK_DIR" "$COMMON_KG_DIR"
cd "$REPO_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

verify_cache() {
  conda run -n zeroddi python -c \
    "import json,sqlite3,sys; c=sqlite3.connect('$KG_FILE'); md=dict(c.execute('select key,value from metadata')); ok=c.execute('pragma integrity_check').fetchone()[0]=='ok' and json.loads(md.get('format','null'))=='molecbionet_pair_graph_v2' and c.execute('select count(*) from pairs').fetchone()[0]==json.loads(md['metadata'])['mapped_pairs']; c.close(); sys.exit(0 if ok else 1)"
}

# All splits share a single pair cache. flock prevents two pipelines from
# rebuilding the same temporary database concurrently.
exec 9>"$COMMON_KG_DIR/molecbionet_pair_graph.lock"
flock 9
if verify_cache; then
  echo "[$(date --iso-8601=seconds)] $SPLIT: shared pair graph cache is valid"
else
  echo "[$(date --iso-8601=seconds)] $SPLIT: building shared pair graph cache"
  conda run -n zeroddi python tools/build_molecbionet_pair_graphs.py \
    --zeroddi-csv data/KnowDDI/drugbank_true_s0/DDI_final1.5.csv \
    --split-files data/KnowDDI/drugbank_true_s0/DDI_final1.5.csv \
    --molecbionet-dir data/MolecBioNet/DrugBank \
    --max-nodes 256 \
    --max-edges 512 \
    --output "$KG_FILE" \
    --stats-output "$COMMON_KG_DIR/molecbionet_pair_graph_s0_stats.json"
  verify_cache
fi
flock -u 9

wait_for_gpu() {
  while true; do
    local free_mib
    free_mib=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 30000 )); then
      return
    fi
    echo "[$(date --iso-8601=seconds)] $SPLIT: GPU $GPU_ID has ${free_mib:-unknown} MiB free; waiting"
    sleep 60
  done
}

if [[ -f "$TRAIN_MARKER" && -f "$CHECKPOINT" ]]; then
  echo "[$(date --iso-8601=seconds)] $SPLIT: training marker and checkpoint exist; skipping training"
else
  wait_for_gpu
  echo "[$(date --iso-8601=seconds)] $SPLIT: starting training on physical GPU $GPU_ID"
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run -n zeroddi python main.py \
    --config "$CONFIG" --device cuda:0 --seednumber 42
  test -f "$CHECKPOINT"
  touch "$TRAIN_MARKER"
  echo "[$(date --iso-8601=seconds)] $SPLIT: training completed"
fi

if [[ -f "$EVAL_MARKER" ]]; then
  echo "[$(date --iso-8601=seconds)] $SPLIT: test evaluation marker exists; skipping evaluation"
else
  wait_for_gpu
  echo "[$(date --iso-8601=seconds)] $SPLIT: evaluating best checkpoint on physical GPU $GPU_ID"
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run -n zeroddi python main.py \
    --config "$CONFIG" --device cuda:0 --seednumber 42 --seen_para "$CHECKPOINT" \
    2>&1 | tee -a "$WORK_DIR/test_eval_console.log"
  touch "$EVAL_MARKER"
  echo "[$(date --iso-8601=seconds)] $SPLIT: evaluation completed"
fi

echo "[$(date --iso-8601=seconds)] $SPLIT: pipeline completed"
