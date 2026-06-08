#!/bin/bash
set -euo pipefail

# Original VarGrad/FairGrad Cityscapes launcher.
# Runs trainer_vargrad.py with VarGrad preprocessing and FairGrad every-step updates.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

METHOD="${METHOD:-fairgrad}"
ALPHA="${ALPHA:-2.0}"
GAMMA="${GAMMA:-1e-5}"
BETA="${BETA:-0.85}"
WEIGHTS_THRESHOLD="${WEIGHTS_THRESHOLD:-1.5}"
USE_THRESHOLD="${USE_THRESHOLD:-false}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-4}"
MODEL="${MODEL:-mtan}"
N_STEPS="${N_STEPS:-1}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/dataset/cityscapes2}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-/root/autodl-tmp/exp_logs_save/vargrad_original/cityscapes}"
run_stamp="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
round_name="${ROUND_NAME:-round_${run_stamp}}"
EXP_ROOT="${EXP_ROOT:-${BASE_OUTPUT_ROOT}/rounds/${round_name}}"
LOG_DIR="${LOG_DIR:-${EXP_ROOT}/log}"
SAVE_DIR="${SAVE_DIR:-${EXP_ROOT}/save}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${EXP_ROOT}/archive}"
SEEDS=( ${SEEDS:-0 1 2} )
GPUS=( ${GPUS:-0 1 2} )

if [ "${#SEEDS[@]}" -ne "${#GPUS[@]}" ]; then
    echo "SEEDS and GPUS must have the same length" >&2
    exit 1
fi
if [ ! -d "$DATA_PATH" ]; then
    echo "DATA_PATH does not exist: $DATA_PATH" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$SAVE_DIR" "$ARCHIVE_DIR"

archive_existing() {
    shopt -s nullglob
    local stamp archive
    stamp=$(date +"%Y%m%d_%H%M%S")
    archive="${ARCHIVE_DIR}/rerun_original_vargrad_${METHOD}_every_step_${stamp}"
    local files=(
        "${LOG_DIR}"/original_vargrad_"${METHOD}"_beta"${BETA}"_alpha"${ALPHA}"_every_step_sd*.log
        "${SAVE_DIR}"/"${METHOD}"_alpha"${ALPHA}"_always"${WEIGHTS_THRESHOLD}"_sd*_N"${N_STEPS}"_bs"${BATCH_SIZE}".stats
    )
    if [ "${#files[@]}" -gt 0 ]; then
        mkdir -p "$archive"
        mv "${files[@]}" "$archive"/
        echo "Archived previous matching artifacts to $archive"
    fi
    shopt -u nullglob
}

archive_existing

echo "Starting original VarGrad ${METHOD} every_step Cityscapes experiments"
echo "DATA_PATH=$DATA_PATH"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SAVE_DIR=$SAVE_DIR"
echo "METHOD=$METHOD ALPHA=$ALPHA BETA=$BETA USE_THRESHOLD=$USE_THRESHOLD BATCH_SIZE=$BATCH_SIZE EPOCHS=$EPOCHS LR=$LR MODEL=$MODEL PYTHON_BIN=$PYTHON_BIN"

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    gpu="${GPUS[$i]}"
    log_file="${LOG_DIR}/original_vargrad_${METHOD}_beta${BETA}_alpha${ALPHA}_every_step_sd${seed}.log"
    cmd=(
        "$PYTHON_BIN" -u trainer_vargrad.py
        --method "$METHOD"
        --alpha "$ALPHA"
        --gamma "$GAMMA"
        --beta "$BETA"
        --weights_threshold "$WEIGHTS_THRESHOLD"
        --use_threshold "$USE_THRESHOLD"
        --seed "$seed"
        --gpu "$gpu"
        --batch-size "$BATCH_SIZE"
        --n-epochs "$EPOCHS"
        --lr "$LR"
        --N_steps "$N_STEPS"
        --model "$MODEL"
        --data-path "$DATA_PATH"
        --save-dir "$SAVE_DIR"
        --preprocessing vargrad
        --scheduler every_step
        --use-vargrad true
        --use-psmgd false
    )
    echo "Launching seed=$seed gpu=$gpu"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '
Log file: %s
' "$log_file"
    nohup "${cmd[@]}" > "$log_file" 2>&1 &
    echo "PID=$!"
    sleep 5
done

echo "All experiments started."
