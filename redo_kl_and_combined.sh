#!/bin/bash

mkdir -p logs

# --- Set 1: entropy ratio ∈ [0.6, 0.7, 0.9], discard-kl ∈ [0.01, 0.05, 0.1] ---
for RATIO in 0.5; do
  for THRESHOLD in 0.01 0.05 0.1; do
    OUTDIR="../experiments/beam_search/kl_discard_${THRESHOLD}_entropy_ratio_${RATIO}"
    python -um main.scripts.basecall \
      --path ../data/goldman_synthetic_data/ \
      --checkpoint ../checkpoint_570209.pt \
      --output $OUTDIR \
      --chunk-size 2000 --window-overlap 0 --batch-size 64 \
      --beam-size 3 --beam-threshold 0.0 --verbose --overwrite \
      --decoder beam_search --model SACallLoRA --is-synthetic \
      --entropy-ratio $RATIO --discard-kl $THRESHOLD \
      | tee -a logs/beam_search.log

    python -um main.scripts.evaluate_model \
      --reference ../data/goldman_synthetic_data \
      --predictions $OUTDIR \
      --verbose --is-synthetic \
      | tee -a logs/beam_search.log
  done
done

# # --- Set 2: desired-length fixed at 117, max-repeats is 1, LENGTH penalty lambda [5 10 15], KL penalty THETA [1 5 10]
# for LAMBDA in 5 10 15; do
#   for THETA in 1 5 10; do
#     OUTDIR="../experiments/beam_search/combined_no_repeats_length_lambda_${LAMBDA}_kl_penalty_${THETA}"
#     python -um main.scripts.basecall \
#         --path ../data/goldman_synthetic_data/ \
#         --checkpoint ../checkpoint_570209.pt \
#         --output $OUTDIR \
#         --chunk-size 2000 --window-overlap 0 --batch-size 64 \
#         --beam-size 3 --beam-threshold 0.0 --verbose --overwrite \
#         --decoder beam_search --model SACallLoRA --is-synthetic \
#         --desired-length 117 --max-repeats 1 \
#         --kl-lambda $THETA --length-lambda $LAMBDA \
#         | tee -a logs/beam_search.log

#     python -um main.scripts.evaluate_model \
#         --reference ../data/goldman_synthetic_data \
#         --predictions $OUTDIR \
#         --verbose --is-synthetic \
#         | tee -a logs/beam_search.log
#     done
# done