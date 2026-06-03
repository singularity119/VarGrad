#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/VarGrad/VarGrad-code}"
LAUNCHER="$REPO_ROOT/experiments/celeba/run_vargrad_fairgrad_psmgd.sh"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/exp_logs_save/vargrad_reimpl/celeba/save}"
LOG_ROOT="${LOG_ROOT:-/root/autodl-tmp/exp_logs_save/vargrad_reimpl/celeba/log}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/exp_logs_save/vargrad_reimpl/celeba/batch_runs}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

mkdir -p "$SAVE_DIR" "$LOG_ROOT" "$RUN_ROOT"

RUN_ID="${RUN_ID:-celeba_ordered_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
ARCHIVE_DIR="$RUN_DIR/archive_existing"
BATCH_LOG="$RUN_DIR/batch.log"
STATE_FILE="$RUN_DIR/state.tsv"
SUMMARY_FILE="$RUN_DIR/summary.tsv"
mkdir -p "$RUN_DIR" "$ARCHIVE_DIR"

exec > >(tee -a "$BATCH_LOG") 2>&1

echo "[batch] run_id=$RUN_ID"
echo "[batch] started_at=$(date -Is)"
echo "[batch] repo_root=$REPO_ROOT"
echo "[batch] save_dir=$SAVE_DIR"
echo "[batch] log_root=$LOG_ROOT"

if [[ ! -x "$LAUNCHER" ]]; then
  chmod +x "$LAUNCHER"
fi

stem_for() {
  local preprocessing="$1"
  local scheduler="$2"
  local seed="$3"
  local r="$4"
  local metric="$5"
  local direction="$6"
  local threshold="$7"
  local beta="0.85"
  local alpha="2.0"
  local psmgd_alpha="0.5"

  if [[ "$scheduler" == "psmgd_periodic" ]]; then
    printf "vargrad_reimpl_%s_beta%s_fairgrad_alpha%s_psmgd_R%s_a%s_sd%s" \
      "$preprocessing" "$beta" "$alpha" "$r" "$psmgd_alpha" "$seed"
  elif [[ "$scheduler" == "psmgd_dynamic" ]]; then
    printf "vargrad_reimpl_%s_beta%s_fairgrad_alpha%s_psmgd_dynamic_%s_%s_thr%s_a%s_sd%s" \
      "$preprocessing" "$beta" "$alpha" "$metric" "$direction" "$threshold" "$psmgd_alpha" "$seed"
  else
    printf "vargrad_reimpl_%s_beta%s_fairgrad_alpha%s_%s_sd%s" \
      "$preprocessing" "$beta" "$alpha" "$scheduler" "$seed"
  fi
}

trainer_pids() {
  ps -ww -eo pid=,cmd= \
    | awk '/[p]ython/ && /trainer.py/ && /--data-path/ && /celeba/ {print $1}' \
    | sort -n
}

trainer_pid_running() {
  local pid="$1"
  ps -p "$pid" -o cmd= 2>/dev/null | grep -q 'trainer.py'
}

archive_existing() {
  local group="$1"
  local preprocessing="$2"
  local scheduler="$3"
  local r="$4"
  local metric="$5"
  local direction="$6"
  local threshold="$7"
  local seed stem target group_archive
  group_archive="$ARCHIVE_DIR/$group"
  mkdir -p "$group_archive"

  for seed in 0 1 2; do
    stem="$(stem_for "$preprocessing" "$scheduler" "$seed" "$r" "$metric" "$direction" "$threshold")"
    for target in \
      "$LOG_ROOT/${stem}.log" \
      "$SAVE_DIR/${stem}.stats" \
      "$SAVE_DIR/${stem}.u_telemetry.jsonl"; do
      if [[ -e "$target" ]]; then
        echo "[archive] moving existing $target -> $group_archive/"
        mv "$target" "$group_archive/"
      fi
    done
  done
}

launch_one() {
  local group="$1"
  local preprocessing="$2"
  local scheduler="$3"
  local r="$4"
  local metric="$5"
  local direction="$6"
  local threshold="$7"
  local save_telemetry="$8"
  local seed="$9"
  local gpu="${10}"
  local before after pid stem log_file attempts

  stem="$(stem_for "$preprocessing" "$scheduler" "$seed" "$r" "$metric" "$direction" "$threshold")"
  log_file="$LOG_ROOT/${stem}.log"
  before="$(mktemp)"
  after="$(mktemp)"
  trainer_pids > "$before"

  echo "[launch] group=$group seed=$seed gpu=$gpu stem=$stem"
  CUDA_VISIBLE_DEVICES="$gpu" \
  METHOD="fairgrad" \
  SOLVER="fairgrad" \
  PREPROCESSING="$preprocessing" \
  SCHEDULER="$scheduler" \
  SEED="$seed" \
  BETA="0.85" \
  ALPHA="2.0" \
  PSMGD_R="$r" \
  PSMGD_ALPHA="0.5" \
  PSMGD_DYNAMIC_METRIC="$metric" \
  PSMGD_DYNAMIC_DIRECTION="$direction" \
  PSMGD_DYNAMIC_THRESHOLD="$threshold" \
  SAVE_U_TELEMETRY="$save_telemetry" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash "$LAUNCHER"

  pid=""
  for attempts in $(seq 1 30); do
    sleep 1
    trainer_pids > "$after"
    pid="$(awk 'NR==FNR {seen[$1]=1; next} !seen[$1] {print $1}' "$before" "$after" | head -1)"
    if [[ -n "$pid" ]]; then
      break
    fi
  done

  rm -f "$before" "$after"

  if [[ -z "$pid" ]]; then
    echo "[error] failed to detect trainer PID for group=$group seed=$seed log=$log_file"
    [[ -f "$log_file" ]] && tail -80 "$log_file" || true
    exit 1
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$group" "$seed" "$gpu" "$pid" "$stem" "$log_file" >> "$STATE_FILE"
  echo "[launch] detected_pid=$pid group=$group seed=$seed"
}

wait_group() {
  local group="$1"
  local pids_csv="$2"
  local pid
  IFS=',' read -r -a pids <<< "$pids_csv"

  echo "[wait] group=$group pids=$pids_csv"
  while true; do
    local running=0
    for pid in "${pids[@]}"; do
      if trainer_pid_running "$pid"; then
        running=$((running + 1))
      fi
    done
    if [[ "$running" -eq 0 ]]; then
      break
    fi
    echo "[wait] $(date -Is) group=$group running=$running"
    sleep 300
  done
}

verify_group() {
  local group="$1"
  local preprocessing="$2"
  local scheduler="$3"
  local r="$4"
  local metric="$5"
  local direction="$6"
  local threshold="$7"
  local seed stem log_file stats_file

  for seed in 0 1 2; do
    stem="$(stem_for "$preprocessing" "$scheduler" "$seed" "$r" "$metric" "$direction" "$threshold")"
    log_file="$LOG_ROOT/${stem}.log"
    stats_file="$SAVE_DIR/${stem}.stats"
    if [[ ! -f "$log_file" ]]; then
      echo "[error] missing log for $stem: $log_file"
      exit 1
    fi
    if ! grep -q "Final Performance" "$log_file"; then
      echo "[error] log lacks Final Performance for $stem"
      tail -120 "$log_file" || true
      exit 1
    fi
    if [[ ! -f "$stats_file" ]]; then
      echo "[error] missing stats for $stem: $stats_file"
      exit 1
    fi
    echo "[verify] ok group=$group seed=$seed stem=$stem"
  done
}

run_group() {
  local group="$1"
  local preprocessing="$2"
  local scheduler="$3"
  local r="$4"
  local metric="$5"
  local direction="$6"
  local threshold="$7"
  local save_telemetry="$8"
  local pids

  echo "[group] start name=$group preprocessing=$preprocessing scheduler=$scheduler R=$r metric=$metric direction=$direction threshold=$threshold telemetry=$save_telemetry"
  archive_existing "$group" "$preprocessing" "$scheduler" "$r" "$metric" "$direction" "$threshold"
  launch_one "$group" "$preprocessing" "$scheduler" "$r" "$metric" "$direction" "$threshold" "$save_telemetry" 0 0
  launch_one "$group" "$preprocessing" "$scheduler" "$r" "$metric" "$direction" "$threshold" "$save_telemetry" 1 1
  launch_one "$group" "$preprocessing" "$scheduler" "$r" "$metric" "$direction" "$threshold" "$save_telemetry" 2 2
  pids="$(awk -F '\t' -v group="$group" '$1 == group {print $4}' "$STATE_FILE" | paste -sd, -)"
  wait_group "$group" "$pids"
  verify_group "$group" "$preprocessing" "$scheduler" "$r" "$metric" "$direction" "$threshold"
  echo "[group] complete name=$group at=$(date -Is)"
}

write_summary() {
  "$PYTHON_BIN" - "$SAVE_DIR" "$SUMMARY_FILE" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import torch

save_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

groups = [
    ("vargrad+fairgrad+every_step", "vargrad_reimpl_vargrad_beta0.85_fairgrad_alpha2.0_every_step_sd{}", None),
    ("vargrad+fairgrad+psmgd_dynamic_above10%", "vargrad_reimpl_vargrad_beta0.85_fairgrad_alpha2.0_psmgd_dynamic_step_rel_fro_above_thr1.76_a0.5_sd{}", "dynamic"),
    ("vargrad+fairgrad+psmgd_periodic_R5", "vargrad_reimpl_vargrad_beta0.85_fairgrad_alpha2.0_psmgd_R5_a0.5_sd{}", "R5"),
    ("none+fairgrad+every_step", "vargrad_reimpl_identity_beta0.85_fairgrad_alpha2.0_every_step_sd{}", None),
    ("none+fairgrad+psmgd_dynamic_above10%", "vargrad_reimpl_identity_beta0.85_fairgrad_alpha2.0_psmgd_dynamic_step_rel_fro_above_thr1.76_a0.5_sd{}", "dynamic"),
]

rows = ["group\tseed\tbest_epoch\tdelta_m\tmean_f1\tsolver_calls\tmean_interval\trounded_R\tstats_path\ttelemetry_path"]
for group, pattern, cadence in groups:
    for seed in (0, 1, 2):
        stem = pattern.format(seed)
        stats_path = save_dir / f"{stem}.stats"
        telemetry_path = save_dir / f"{stem}.u_telemetry.jsonl"
        best_epoch = ""
        delta_m = ""
        mean_f1 = ""
        if stats_path.exists():
            stats = torch.load(stats_path, map_location="cpu")
            metric = np.asarray(stats["metric"], dtype=float)
            deltas = np.asarray(stats["delta_m"], dtype=float)
            best_epoch_idx = int(stats["best_epoch"]) if stats.get("best_epoch") is not None else len(deltas) - 1
            best_epoch = best_epoch_idx + 1
            delta_m = f"{float(deltas[best_epoch_idx]):.6f}"
            mean_f1 = f"{float(metric[best_epoch_idx].mean()):.6f}"

        solver_calls = ""
        mean_interval = ""
        rounded_R = "5" if cadence == "R5" else ""
        if cadence == "dynamic" and telemetry_path.exists():
            update_steps = []
            with telemetry_path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("solver_called") and row.get("updated_weights"):
                        update_steps.append(int(row["global_step"]))
            solver_calls = str(len(update_steps))
            if len(update_steps) > 1:
                intervals = np.diff(update_steps)
                mean_interval_value = float(intervals.mean())
                mean_interval = f"{mean_interval_value:.6f}"
                rounded_R = str(int(round(mean_interval_value)))

        rows.append(
            "\t".join(
                map(
                    str,
                    [
                        group,
                        seed,
                        best_epoch,
                        delta_m,
                        mean_f1,
                        solver_calls,
                        mean_interval,
                        rounded_R,
                        stats_path,
                        telemetry_path if telemetry_path.exists() else "",
                    ],
                )
            )
        )

summary_path.write_text("\n".join(rows) + "\n")
print(summary_path)
PY
  cat "$SUMMARY_FILE"
}

: > "$STATE_FILE"
printf "group\tseed\tgpu\tpid\tstem\tlog_file\n" > "$STATE_FILE"

run_group "01_vargrad_fairgrad_every_step" "vargrad" "every_step" "10" "step_rel_fro" "above" "1.76" "false"
run_group "02_vargrad_fairgrad_dynamic_above10" "vargrad" "psmgd_dynamic" "10" "step_rel_fro" "above" "1.76" "true"
run_group "03_vargrad_fairgrad_periodic_R5" "vargrad" "psmgd_periodic" "5" "step_rel_fro" "above" "1.76" "false"
run_group "04_identity_fairgrad_every_step" "identity" "every_step" "10" "step_rel_fro" "above" "1.76" "false"
run_group "05_identity_fairgrad_dynamic_above10" "identity" "psmgd_dynamic" "10" "step_rel_fro" "above" "1.76" "true"

write_summary
echo "[batch] completed_at=$(date -Is)"
