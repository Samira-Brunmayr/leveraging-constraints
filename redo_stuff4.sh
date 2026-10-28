#!/bin/bash

mkdir -p logs


# --- Set 1: discard-kl ∈ [0.01,0.05,0.1] ---
for LAMBDA in 0.01 0.05 0.1; do
  OUTDIR="../experiments/beam_search/kl_discard_${LAMBDA}"
  python -um main.scripts.basecall \
    --path ../data/goldman_synthetic_data/ \
    --checkpoint ../checkpoint_570209.pt \
    --output $OUTDIR \
    --chunk-size 2000 --window-overlap 0 --batch-size 64 \
    --beam-size 3 --beam-threshold 0.0 --verbose --overwrite \
    --decoder beam_search --model SACallLoRA --is-synthetic \
    --discard-kl $LAMBDA \
    | tee -a logs/beam_search.log

  python -um main.scripts.evaluate_model \
    --reference ../data/goldman_synthetic_data \
    --predictions $OUTDIR \
    --verbose --is-synthetic \
    | tee -a logs/beam_search.log
done