#!/bin/bash

mkdir -p logs

# --- Set 1: max-repeats ∈ [1, 2], homopolymer-penalty ∈ [0.2, 0.5] ---
for REPEATS in 1 2; do
  for PENALTY in 0.2 0.5; do
    OUTDIR="../experiments/beam_search/max_repeats_${REPEATS}_length_lambda_${PENALTY}"
    python -um main.scripts.basecall \
      --path ../data/goldman_synthetic_data/ \
      --checkpoint ../checkpoint_570209.pt \
      --output $OUTDIR \
      --chunk-size 2000 --window-overlap 0 --batch-size 64 \
      --beam-size 3 --beam-threshold 0.0 --verbose --overwrite \
      --decoder beam_search --model SACallLoRA --is-synthetic \
      --max-repeats $REPEATS --homopolymer-penalty $PENALTY \
      | tee -a logs/beam_search.log

    python -um main.scripts.evaluate_model \
      --reference ../data/goldman_synthetic_data \
      --predictions $OUTDIR \
      --verbose --is-synthetic \
      | tee -a logs/beam_search.log
  done
done

# --- Set 2: max-repeats ∈ [1, 2] ---
for REPEATS in 1 2; do
  OUTDIR="../experiments/beam_search/max_repeats_${REPEATS}"
  python -um main.scripts.basecall \
    --path ../data/goldman_synthetic_data/ \
    --checkpoint ../checkpoint_570209.pt \
    --output $OUTDIR \
    --chunk-size 2000 --window-overlap 0 --batch-size 64 \
    --beam-size 3 --beam-threshold 0.0 --verbose --overwrite \
    --decoder beam_search --model SACallLoRA --is-synthetic \
    --max-repeats $REPEATS \
    | tee -a logs/beam_search.log

  python -um main.scripts.evaluate_model \
    --reference ../data/goldman_synthetic_data \
    --predictions $OUTDIR \
    --verbose --is-synthetic \
    | tee -a logs/beam_search.log
done

# --- Set 3: desired-length fixed at 117, length-lambda ∈ [1, 5, 10, 15, 20] ---
for LAMBDA in 1 5 10 15 20; do
  OUTDIR="../experiments/beam_search/length_lambda_${LAMBDA}"
  python -um main.scripts.basecall \
    --path ../data/goldman_synthetic_data/ \
    --checkpoint ../checkpoint_570209.pt \
    --output $OUTDIR \
    --chunk-size 2000 --window-overlap 0 --batch-size 64 \
    --beam-size 3 --beam-threshold 0.0 --verbose --overwrite \
    --decoder beam_search --model SACallLoRA --is-synthetic \
    --desired-length 117 --length-lambda $LAMBDA \
    | tee -a logs/beam_search.log

  python -um main.scripts.evaluate_model \
    --reference ../data/goldman_synthetic_data \
    --predictions $OUTDIR \
    --verbose --is-synthetic \
    | tee -a logs/beam_search.log
done