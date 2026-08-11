#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/wumengying/ZeroDDI"
KG_DIR="$REPO_DIR/data/KnowDDI/drugbank_true_s0/kg"
LOG_DIR="$REPO_DIR/work_dirs/new_s0_reverse_kg_gnn"
PIPELINE_LOG="$LOG_DIR/pipeline.log"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

KG_FILE="$KG_DIR/molecbionet_pair_graph_s0.sqlite"
if conda run -n zeroddi python -c \
  "import sqlite3,sys; c=sqlite3.connect('$KG_FILE'); ok=c.execute('pragma integrity_check').fetchone()[0]=='ok' and c.execute('select count(*) from pairs').fetchone()[0]==603353; c.close(); sys.exit(0 if ok else 1)"; then
  echo "[$(date --iso-8601=seconds)] Existing pair graph is complete; skipping build"
else
  echo "[$(date --iso-8601=seconds)] Building explicit pair graphs"
  conda run -n zeroddi python tools/build_molecbionet_pair_graphs.py \
    --zeroddi-csv data/KnowDDI/drugbank_true_s0/DDI_final1.5.csv \
    --split-files \
      data/KnowDDI/drugbank_true_s0/train.csv \
      data/KnowDDI/drugbank_true_s0/val.csv \
      data/KnowDDI/drugbank_true_s0/test.csv \
    --molecbionet-dir data/MolecBioNet/DrugBank \
    --max-nodes 256 \
    --max-edges 512 \
    --output "$KG_FILE" \
    --stats-output "$KG_DIR/molecbionet_pair_graph_s0_stats.json"
  echo "[$(date --iso-8601=seconds)] Pair graph build completed"
fi
while true; do
  FREE_MIB=$(nvidia-smi --id=4 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$FREE_MIB" =~ ^[0-9]+$ ]] && (( FREE_MIB >= 30000 )); then
    break
  fi
  echo "[$(date --iso-8601=seconds)] GPU 4 has ${FREE_MIB:-unknown} MiB free; waiting for 30000 MiB"
  sleep 60
done

echo "[$(date --iso-8601=seconds)] Starting training on physical GPU 4"
CUDA_VISIBLE_DEVICES=4 conda run -n zeroddi python main.py \
  --config configs/new_s0_reverse_kg_gnn.py \
  --device cuda:0 \
  --seednumber 42
echo "[$(date --iso-8601=seconds)] Pipeline completed"
