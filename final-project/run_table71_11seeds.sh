#!/usr/bin/env bash
set -euo pipefail

# Run this script from the same project directory as your original script.

export CKPT="${CKPT:-../checkpoint_570209.pt}"
export INPUT_DIR="${INPUT_DIR:-./goldman_data}"
export CODE="${CODE:-./codes/code.cd}"
export DATA_ROOT="${DATA_ROOT:-../data}"
export EXP_ROOT="${EXP_ROOT:-../experiments/table71_repeats}"
export THREADS="${THREADS:-10}"

# At most four seed workflows run concurrently, sharing the GPU.
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
SEEDS=(1 2 3 4 5 6 7 8 9 10 42)

for setting in PARALLEL_JOBS BATCH_SIZE THREADS; do
  [[ "${!setting}" =~ ^[1-9][0-9]*$ ]] || {
    echo "$setting must be a positive integer." >&2
    exit 1
  }
done

# Validate required inputs before deleting any previous results.
[[ -d "$INPUT_DIR" ]] || {
  echo "Input directory not found: $INPUT_DIR" >&2
  exit 1
}
for required_file in "$CKPT" "$CODE"; do
  [[ -f "$required_file" ]] || {
    echo "Required file not found: $required_file" >&2
    exit 1
  }
done

mkdir -p -- "$EXP_ROOT/logs"

evaluate_variant() {
  local dataset="$1" outdir="$2"
  shift 2

  printf '=== Evaluating %s ===\n' "$outdir"
  python -um main.scripts.basecall \
    --path "$dataset" \
    --checkpoint "$CKPT" \
    --output "$outdir" \
    --chunk-size 2000 --window-overlap 0 --batch-size "$BATCH_SIZE" \
    --beam-size 3 --beam-threshold 0.0 \
    --verbose --overwrite \
    --decoder beam_search --model SACallLoRA --is-synthetic \
    "$@"

  python -um main.scripts.evaluate_model \
    --reference "$dataset" \
    --predictions "$outdir" \
    --verbose --is-synthetic

  python -m main.scripts.payload_accuracy_synthetic \
    --input-dir "$outdir" \
    --goldman-dir "$dataset"
}

run_seed() {
  local seed="$1"
  local dataset="$DATA_ROOT/goldman_synthetic_data_seed_${seed}"
  local outroot="$EXP_ROOT/seed_${seed}"
  local f name penalty repeats

  evaluate_variant "$dataset" "$outroot/default"

  for penalty in 0.2 0.5; do
    for repeats in 1 2; do
      evaluate_variant \
        "$dataset" \
        "$outroot/max_repeats_${repeats}_penalty_${penalty}" \
        --homopolymer-penalty "$penalty" \
        --max-repeats "$repeats"
    done
  done

  for penalty in 1; do
    for repeats in 1; do
      evaluate_variant \
        "$dataset" \
        "$outroot/max_repeats_${repeats}_penalty_${penalty}" \
        --homopolymer-penalty "$penalty" \
        --max-repeats "$repeats"
    done
  done
}

run_seed_logged() {
  local seed="$1"
  local logfile="$EXP_ROOT/logs/seed_${seed}.log"

  printf '[START] seed %s -- log: %s\n' "$seed" "$logfile"

  # A fresh Bash process keeps errexit active throughout the seed workflow,
  # even though the parent checks its exit status using an if statement.
  if bash -c 'set -euo pipefail; run_seed "$1"' _ "$seed" \
    > "$logfile" 2>&1; then
    printf '[DONE] seed %s\n' "$seed"
  else
    printf '[FAILED] seed %s -- see %s\n' "$seed" "$logfile" >&2
    return 1
  fi
}

export -f evaluate_variant run_seed run_seed_logged

printf 'Running 11 seeds, at most %s at a time (55 evaluations total).\n' "$PARALLEL_JOBS"
if printf '%s\n' "${SEEDS[@]}" |
  xargs -n 1 -P "$PARALLEL_JOBS" \
    bash -c 'set -euo pipefail; run_seed_logged "$1"' _; then
  echo "=== All 11 seeds and 55 evaluations completed ==="
else
  echo "Some seeds failed. Check the [FAILED] messages and seed logs." >&2
  exit 1
fi
