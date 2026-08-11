#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/wumengying/ZeroDDI"
GPU_ID="${1:-4}"
cd "$REPO_DIR"

bash tools/run_kg_gnn_split_pipeline.sh s1 "$GPU_ID"
bash tools/run_kg_gnn_split_pipeline.sh s2 "$GPU_ID"
