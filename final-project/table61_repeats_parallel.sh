#!/usr/bin/env bash
set -euo pipefail
set -o noclobber

# Reuse existing Goldman datasets. This wrapper contains no deletion or
# dataset-generation commands. Every invocation creates a new results root.
# The three Python modules are external: their internal filesystem behavior
# cannot be confirmed from this wrapper alone.
# Run from the same project directory as your original script.

export BASE_CKPT="${BASE_CKPT:-../checkpoint_570209.pt}"
export DATA_ROOT="${DATA_ROOT:-../data}"
export TRAIN_ROOT="${TRAIN_ROOT:-../experiments/finetune_multiseed}"
export OUT_ROOT="${OUT_ROOT:-../experiments/table61_eval_3train_10test}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_SEEDS=(1 2 3 4 5 6 7 8 9 10 42)
TRAIN_SEEDS=(1 2 42)

# model folder | LoRA rank (empty for whole-model and decoder fine-tuning)
export MODEL_SPECS="5000_bacterial_benchmark_finetuned_45000|
5000_images_model_decoder_finetune_45000|
5000_images_model_finetune_rank_1_45000|1
5000_images_model_finetune_rank_2_45000|2
5000_images_model_finetune_rank_4_45000|4
5000_images_model_finetune_rank_8_45000|8
5000_images_model_finetune_rank_16_45000|16"

for setting in PARALLEL_JOBS BATCH_SIZE; do
  [[ "${!setting}" =~ ^[1-9][0-9]*$ ]] || {
    echo "$setting must be a positive integer." >&2
    exit 1
  }
done

# Validate all 11 datasets and 22 checkpoints before creating any results.
[[ -f "$BASE_CKPT" ]] || {
  echo "Baseline checkpoint not found: $BASE_CKPT" >&2
  exit 1
}
for test_seed in "${TEST_SEEDS[@]}"; do
  dataset="$DATA_ROOT/goldman_synthetic_data_seed_${test_seed}"
  [[ -d "$dataset" ]] || {
    echo "Dataset not found: $dataset" >&2
    exit 1
  }
done
for train_seed in "${TRAIN_SEEDS[@]}"; do
  while IFS='|' read -r model rank; do
    ckpt="$TRAIN_ROOT/seed_${train_seed}/$model/checkpoints/checkpoint_45000.pt"
    [[ -f "$ckpt" ]] || {
      echo "Checkpoint not found: $ckpt" >&2
      exit 1
    }
  done <<< "$MODEL_SPECS"
done

# mktemp creates a directory that did not previously exist. No old run is
# reused, overwritten, or cleaned up, including when a run fails.
mkdir -p -- "$OUT_ROOT"
export RUN_ROOT
RUN_ROOT=$(mktemp -d "$OUT_ROOT/run_$(date -u '+%Y%m%dT%H%M%SZ')_XXXXXX")
mkdir -p -- "$RUN_ROOT/logs"
printf 'Results directory: %s\n' "$RUN_ROOT"
{
  printf 'BASE_CKPT=%s\nDATA_ROOT=%s\nTRAIN_ROOT=%s\n' \
    "$BASE_CKPT" "$DATA_ROOT" "$TRAIN_ROOT"
  printf 'PARALLEL_JOBS=%s\nBATCH_SIZE=%s\n' "$PARALLEL_JOBS" "$BATCH_SIZE"
  printf 'TRAIN_SEEDS=1 2 42\nTEST_SEEDS=1 2 3 4 5 6 7 8 9 10 42\n'
  printf 'EVALUATIONS=242\n'
} > "$RUN_ROOT/run_info.txt"

evaluate_variant() {
  local dataset="$1" ckpt="$2" outdir="$3"
  shift 3

  # Refuse an existing output path rather than replacing its contents.
  if [[ -e "$outdir" || -L "$outdir" ]]; then
    echo "Refusing to reuse an existing output path: $outdir" >&2
    return 1
  fi
  mkdir -p -- "$(dirname "$outdir")"

  printf '=== Evaluating %s ===\n' "$outdir"
  python -um main.scripts.basecall \
    --path "$dataset" \
    --checkpoint "$ckpt" \
    --output "$outdir" \
    --chunk-size 2000 --window-overlap 0 --batch-size "$BATCH_SIZE" \
    --beam-size 3 --beam-threshold 0.0 \
    --verbose \
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

run_test_seed() {
  local test_seed="$1"
  local dataset="$DATA_ROOT/goldman_synthetic_data_seed_${test_seed}"
  local train_seed model rank ckpt outdir
  local -a lora_args=()

  evaluate_variant "$dataset" "$BASE_CKPT" \
    "$RUN_ROOT/baseline/test_${test_seed}/default"

  for train_seed in 1 2 42; do
    # A separate descriptor keeps Python stdin away from the model list.
    while IFS='|' read -r -u 3 model rank; do
      lora_args=()
      if [[ -n "$rank" ]]; then
        lora_args=(--lora-rank "$rank" -q -k)
      fi
      ckpt="$TRAIN_ROOT/seed_${train_seed}/$model/checkpoints/checkpoint_45000.pt"
      outdir="$RUN_ROOT/train_${train_seed}/test_${test_seed}/$model"
      evaluate_variant "$dataset" "$ckpt" "$outdir" "${lora_args[@]}"
    done 3<<< "$MODEL_SPECS"
  done
}

run_test_seed_logged() {
  local test_seed="$1"
  local logfile="$RUN_ROOT/logs/test_${test_seed}.log"

  printf '[START] test seed %s -- log: %s\n' "$test_seed" "$logfile"
  if bash -c 'set -euo pipefail; set -o noclobber; run_test_seed "$1"' \
    _ "$test_seed" > "$logfile" 2>&1; then
    printf '[DONE] test seed %s\n' "$test_seed"
  else
    printf '[FAILED] test seed %s -- see %s\n' "$test_seed" "$logfile" >&2
    return 1
  fi
}

export -f evaluate_variant run_test_seed run_test_seed_logged

# Each dataset is used by one worker at a time. Within that worker, its
# baseline and 21 fine-tuned model evaluations run sequentially.
printf 'Running 242 evaluations, at most %s test seeds at a time.\n' "$PARALLEL_JOBS"
if printf '%s\n' "${TEST_SEEDS[@]}" |
  xargs -n 1 -P "$PARALLEL_JOBS" \
    bash -c 'set -euo pipefail; set -o noclobber; run_test_seed_logged "$1"' _; then
  printf '=== All 242 evaluations completed. Results: %s ===\n' "$RUN_ROOT"
else
  printf 'Some test seeds failed. Results directory: %s\n' "$RUN_ROOT" >&2
  exit 1
fi
