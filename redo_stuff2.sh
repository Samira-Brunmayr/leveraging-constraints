#!/bin/bash

mkdir -p logs


# --- Set 1: desired-length fixed at 117, discard-expected-length ∈ [0.05,0.1,0.2,0.5] ---
for THETA in 0.05 0.1 0.2 0.5; do
  OUTDIR="../experiments/beam_search/length_discard_${THETA}"
  python -um main.scripts.basecall \
    --path ../data/goldman_synthetic_data/ \
    --checkpoint ../checkpoint_570209.pt \
    --output $OUTDIR \
    --chunk-size 2000 --window-overlap 0 --batch-size 64 \
    --beam-size 3 --beam-threshold 0.0 --verbose --overwrite \
    --decoder beam_search --model SACallLoRA --is-synthetic \
    --desired-length 117 --discard-expected-length $THETA \
    | tee -a logs/beam_search.log

  python -um main.scripts.evaluate_model \
    --reference ../data/goldman_synthetic_data \
    --predictions $OUTDIR \
    --verbose --is-synthetic \
    | tee -a logs/beam_search.log
done