#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/home/wumengying/ZeroDDI
PYTHON_BIN=/home/wumengying/miniconda3/envs/zeroddi/bin/python
CONFIG=configs/new_s2_mechanism_m2_pairwise.py
RUN_DIR=work_dirs/mechanism_m2_s2_seed42
PHYSICAL_GPU=${1:-7}
SEED=42
EPOCHS=100
CHECKPOINT="$RUN_DIR/model_parameter/model_best_epoch${EPOCHS}_seen${SEED}.pkl"

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$RUN_DIR"

{
    echo "[$(date '+%F %T')] START train M2-S2 on physical GPU $PHYSICAL_GPU"
    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        "$PYTHON_BIN" main.py --config "$CONFIG" --work-dir "$RUN_DIR" \
        --device cuda:0 --seednumber "$SEED" --max-epochs "$EPOCHS"
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "[$(date '+%F %T')] FAILED train M2-S2 status=$status"
        exit "$status"
    fi
    if [ ! -f "$CHECKPOINT" ]; then
        echo "[$(date '+%F %T')] FAILED missing checkpoint $CHECKPOINT"
        exit 20
    fi

    echo "[$(date '+%F %T')] START test M2-S2 checkpoint=$CHECKPOINT"
    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        "$PYTHON_BIN" main.py --config "$CONFIG" --work-dir "$RUN_DIR" \
        --device cuda:0 --seednumber "$SEED" --max-epochs "$EPOCHS" \
        --seen_para "$CHECKPOINT"
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "[$(date '+%F %T')] FAILED test M2-S2 status=$status"
        exit "$status"
    fi
    echo "[$(date '+%F %T')] COMPLETE train+test M2-S2"
} 2>&1 | tee -a "$RUN_DIR/pipeline.log"

exit "${PIPESTATUS[0]}"
