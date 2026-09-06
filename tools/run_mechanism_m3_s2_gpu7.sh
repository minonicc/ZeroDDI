#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT=/home/wumengying/ZeroDDI
PYTHON_BIN=/home/wumengying/miniconda3/envs/zeroddi/bin/python
CONFIG_PATH=configs/new_s2_mechanism_m3_pairwise_gates.py
RUN_DIR=work_dirs/mechanism_m3_s2_seed42
PHYSICAL_GPU=7
SEED=42
EPOCHS=100
CHECKPOINT="$RUN_DIR/model_parameter/model_best_epoch${EPOCHS}_seen${SEED}.pkl"

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$RUN_DIR"

{
    echo "[$(date '+%F %T')] START train M3-S2 on shared physical GPU $PHYSICAL_GPU"
    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    "$PYTHON_BIN" main.py \
        --config "$CONFIG_PATH" \
        --work-dir "$RUN_DIR" \
        --device cuda:0 \
        --seednumber "$SEED" \
        --max-epochs "$EPOCHS"
    train_status=$?
    if [ "$train_status" -ne 0 ]; then
        echo "[$(date '+%F %T')] FAILED train M3-S2 status=$train_status"
        exit "$train_status"
    fi
    if [ ! -f "$CHECKPOINT" ]; then
        echo "[$(date '+%F %T')] FAILED missing checkpoint $CHECKPOINT"
        exit 20
    fi

    echo "[$(date '+%F %T')] START test M3-S2 checkpoint=$CHECKPOINT"
    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    "$PYTHON_BIN" main.py \
        --config "$CONFIG_PATH" \
        --work-dir "$RUN_DIR" \
        --device cuda:0 \
        --seednumber "$SEED" \
        --max-epochs "$EPOCHS" \
        --seen_para "$CHECKPOINT"
    test_status=$?
    if [ "$test_status" -ne 0 ]; then
        echo "[$(date '+%F %T')] FAILED test M3-S2 status=$test_status"
        exit "$test_status"
    fi
    echo "[$(date '+%F %T')] COMPLETE train+test M3-S2"
} 2>&1 | tee -a "$RUN_DIR/pipeline.log"

exit "${PIPESTATUS[0]}"
