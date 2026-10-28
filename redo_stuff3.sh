#!/bin/bash

mkdir -p logs


# --- Set 1: kl-lambda ∈ [1,5,10,15,20] ---
for LAMBDA in 1 5 10 15 20; do
  OUTDIR="../experiments/beam_search/kl_penalty_${LAMBDA}"
  python -um main.scripts.basecall \
    --path ../data/goldman_synthetic_data/ \
    --checkpoint ../checkpoint_570209.pt \
    --output $OUTDIR \
    --chunk-size 2000 --window-overlap 0 --batch-size 64 \
    --beam-size 3 --beam-threshold 0.0 --verbose --overwrite \
    --decoder beam_search --model SACallLoRA --is-synthetic \
    --kl-lambda $LAMBDA \
    | tee -a logs/beam_search.log

  python -um main.scripts.evaluate_model \
    --reference ../data/goldman_synthetic_data \
    --predictions $OUTDIR \
    --verbose --is-synthetic \
    | tee -a logs/beam_search.log
done