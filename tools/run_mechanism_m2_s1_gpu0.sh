#!/usr/bin/env bash
set -uo pipefail

cd /home/wumengying/ZeroDDI || exit 1
PYTHON_BIN=/home/wumengying/miniconda3/envs/zeroddi/bin/python
CONFIG=configs/new_s1_mechanism_m2_pairwise.py
RUN_DIR=work_dirs/mechanism_m2_s1_seed42
CHECKPOINT="$RUN_DIR/model_parameter/model_best_epoch100_seen42.pkl"
mkdir -p "$RUN_DIR"

{
    echo "[$(date '+%F %T')] START train M2-S1 on physical GPU 0"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        "$PYTHON_BIN" main.py --config "$CONFIG" --work-dir "$RUN_DIR" \
        --device cuda:0 --seednumber 42 --max-epochs 100
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "[$(date '+%F %T')] FAILED train M2-S1 status=$status"
        exit "$status"
    fi
    if [ ! -f "$CHECKPOINT" ]; then
        echo "[$(date '+%F %T')] FAILED missing checkpoint $CHECKPOINT"
        exit 20
    fi
    echo "[$(date '+%F %T')] START test M2-S1 checkpoint=$CHECKPOINT"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        "$PYTHON_BIN" main.py --config "$CONFIG" --work-dir "$RUN_DIR" \
        --device cuda:0 --seednumber 42 --max-epochs 100 \
        --seen_para "$CHECKPOINT"
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "[$(date '+%F %T')] FAILED test M2-S1 status=$status"
        exit "$status"
    fi
    echo "[$(date '+%F %T')] COMPLETE train+test M2-S1"
} 2>&1 | tee -a "$RUN_DIR/pipeline.log"
exit "${PIPESTATUS[0]}"
