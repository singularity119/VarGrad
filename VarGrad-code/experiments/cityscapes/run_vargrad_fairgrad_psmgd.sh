#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/VarGrad/VarGrad-code}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/dataset/cityscapes2}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-/root/autodl-tmp/exp_logs_save/vargrad_reimpl/cityscapes}"

method="${METHOD:-fairgrad}"
preprocessing="${PREPROCESSING:-vargrad}"
solver="${SOLVER:-fairgrad}"
scheduler="${SCHEDULER:-every_step}"

beta="${BETA:-0.85}"
alpha="${ALPHA:-2.0}"
psmgd_R="${PSMGD_R:-10}"
psmgd_alpha="${PSMGD_ALPHA:-0.5}"
psmgd_dynamic_metric="${PSMGD_DYNAMIC_METRIC:-step_rel_fro}"

if [[ -n "${PSMGD_DYNAMIC_DIRECTION:-}" ]]; then
  psmgd_dynamic_direction="$PSMGD_DYNAMIC_DIRECTION"
elif [[ "$psmgd_dynamic_metric" == "step_rel_fro" ]]; then
  psmgd_dynamic_direction="above"
else
  psmgd_dynamic_direction="below"
fi

if [[ -n "${PSMGD_DYNAMIC_THRESHOLD:-}" ]]; then
  psmgd_dynamic_threshold="$PSMGD_DYNAMIC_THRESHOLD"
else
  case "${psmgd_dynamic_metric}:${psmgd_dynamic_direction}" in
    refresh_rel_fro:below)
      psmgd_dynamic_threshold="1.016"
      ;;
    refresh_rel_fro:above)
      psmgd_dynamic_threshold="1.096"
      ;;
    step_rel_fro:below)
      psmgd_dynamic_threshold="1.12"
      ;;
    step_rel_fro:above)
      psmgd_dynamic_threshold="1.76"
      ;;
    *)
      echo "Unsupported PSMGD dynamic metric/direction: ${psmgd_dynamic_metric}/${psmgd_dynamic_direction}" >&2
      exit 1
      ;;
  esac
fi

seed="${SEED:-0}"
batch_size="${BATCH_SIZE:-8}"
epochs="${EPOCHS:-200}"
lr="${LR:-1e-4}"
model="${MODEL:-mtan}"
save_u_telemetry="${SAVE_U_TELEMETRY:-false}"
post_step_train_forward="${POST_STEP_TRAIN_FORWARD:-false}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python}"

run_stamp="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
round_name="${ROUND_NAME:-round_${run_stamp}}"
RUN_ROOT="${RUN_ROOT:-${BASE_OUTPUT_ROOT}/rounds/${round_name}}"
SAVE_DIR="${SAVE_DIR:-${RUN_ROOT}/save}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/log}"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_ROOT/launch"
cd "$REPO_ROOT/experiments/cityscapes"
export PYTHONPATH="$REPO_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ "$scheduler" == "psmgd_periodic" ]]; then
  run_name="vargrad_reimpl_${preprocessing}_beta${beta}_${solver}_alpha${alpha}_psmgd_R${psmgd_R}_a${psmgd_alpha}_sd${seed}"
elif [[ "$scheduler" == "psmgd_dynamic" ]]; then
  run_name="vargrad_reimpl_${preprocessing}_beta${beta}_${solver}_alpha${alpha}_psmgd_dynamic_${psmgd_dynamic_metric}_${psmgd_dynamic_direction}_thr${psmgd_dynamic_threshold}_a${psmgd_alpha}_sd${seed}"
else
  run_name="vargrad_reimpl_${preprocessing}_beta${beta}_${solver}_alpha${alpha}_${scheduler}_sd${seed}"
fi

log_file="$LOG_ROOT/${run_name}.log"

nohup "$python_bin" -u trainer.py \
  --method "$method" \
  --preprocessing "$preprocessing" \
  --solver "$solver" \
  --scheduler "$scheduler" \
  --beta "$beta" \
  --psmgd-R "$psmgd_R" \
  --psmgd-alpha "$psmgd_alpha" \
  --psmgd-dynamic-threshold "$psmgd_dynamic_threshold" \
  --psmgd-dynamic-metric "$psmgd_dynamic_metric" \
  --psmgd-dynamic-direction "$psmgd_dynamic_direction" \
  --alpha "$alpha" \
  --seed "$seed" \
  --batch-size "$batch_size" \
  --n-epochs "$epochs" \
  --lr "$lr" \
  --model "$model" \
  --data-path "$DATA_ROOT" \
  --save-dir "$SAVE_DIR" \
  --save-u-telemetry "$save_u_telemetry" \
  --post-step-train-forward "$post_step_train_forward" \
  > "$log_file" 2>&1 < /dev/null &

echo "Started Cityscapes run: $log_file"
echo "Run root: $RUN_ROOT"
