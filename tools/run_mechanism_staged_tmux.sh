#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT=/home/wumengying/ZeroDDI
PYTHON_BIN=/home/wumengying/miniconda3/envs/zeroddi/bin/python
SEED=42
EPOCHS=100

cd "$PROJECT_ROOT" || exit 1

run_train_and_test() {
    gpu_id=$1
    config_path=$2
    run_dir=$3
    stage_name=$4
    checkpoint="$run_dir/model_parameter/model_best_epoch${EPOCHS}_seen${SEED}.pkl"

    mkdir -p "$run_dir"
    {
        echo "[$(date '+%F %T')] START train $stage_name on physical GPU $gpu_id"
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        OMP_NUM_THREADS=8 \
        MKL_NUM_THREADS=8 \
        "$PYTHON_BIN" main.py \
            --config "$config_path" \
            --work-dir "$run_dir" \
            --device cuda:0 \
            --seednumber "$SEED" \
            --max-epochs "$EPOCHS"
        train_status=$?
        if [ "$train_status" -ne 0 ]; then
            echo "[$(date '+%F %T')] FAILED train $stage_name status=$train_status"
            return "$train_status"
        fi
        if [ ! -f "$checkpoint" ]; then
            echo "[$(date '+%F %T')] FAILED missing checkpoint $checkpoint"
            return 20
        fi

        echo "[$(date '+%F %T')] START test $stage_name checkpoint=$checkpoint"
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        OMP_NUM_THREADS=8 \
        MKL_NUM_THREADS=8 \
        "$PYTHON_BIN" main.py \
            --config "$config_path" \
            --work-dir "$run_dir" \
            --device cuda:0 \
            --seednumber "$SEED" \
            --max-epochs "$EPOCHS" \
            --seen_para "$checkpoint"
        test_status=$?
        if [ "$test_status" -ne 0 ]; then
            echo "[$(date '+%F %T')] FAILED test $stage_name status=$test_status"
            return "$test_status"
        fi
        echo "[$(date '+%F %T')] COMPLETE train+test $stage_name"
    } 2>&1 | tee -a "$run_dir/pipeline.log"
    return "${PIPESTATUS[0]}"
}

echo "[$(date '+%F %T')] STAGE 1: M1-S0 on GPU0 and M2-S0 on GPU1"
run_train_and_test \
    0 \
    configs/new_s0_mechanism_m1_experts.py \
    work_dirs/mechanism_m1_s0_seed42 \
    M1-S0 &
stage1_gpu0_pid=$!
run_train_and_test \
    1 \
    configs/new_s0_mechanism_m2_pairwise.py \
    work_dirs/mechanism_m2_s0_seed42 \
    M2-S0 &
stage1_gpu1_pid=$!

stage1_status=0
wait "$stage1_gpu0_pid" || stage1_status=$?
wait "$stage1_gpu1_pid" || stage1_status=$?
if [ "$stage1_status" -ne 0 ]; then
    echo "[$(date '+%F %T')] ABORT: stage 1 failed; M3 was not launched"
    exit "$stage1_status"
fi

echo "[$(date '+%F %T')] STAGE 2: M3-S0 on GPU0 and M3-S1 on GPU1"
run_train_and_test \
    0 \
    configs/new_s0_mechanism_m3_pairwise_gates.py \
    work_dirs/mechanism_m3_s0_seed42 \
    M3-S0 &
stage2_gpu0_pid=$!
run_train_and_test \
    1 \
    configs/new_s1_mechanism_m3_pairwise_gates.py \
    work_dirs/mechanism_m3_s1_seed42 \
    M3-S1 &
stage2_gpu1_pid=$!

stage2_status=0
wait "$stage2_gpu0_pid" || stage2_status=$?
wait "$stage2_gpu1_pid" || stage2_status=$?
if [ "$stage2_status" -ne 0 ]; then
    echo "[$(date '+%F %T')] FAILED: stage 2 returned status=$stage2_status"
    exit "$stage2_status"
fi

echo "[$(date '+%F %T')] COMPLETE: all four train+test runs succeeded"
