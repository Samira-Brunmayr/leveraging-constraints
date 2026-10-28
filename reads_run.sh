#!/bin/bash

mkdir -p logs


# --- Set 1: images ∈ [100, 1000, 10000] ---
for READS in 25 50 100; do
  CHECKPOINT_DIR="../experiments/finetune/models/lora_rank_4_${READS}_reads_45000/checkpoints/checkpoint_45000.pt"
  OUTDIR="../experiments/finetune/lora_rank_4_${READS}_reads_45000/"
  python -um main.scripts.basecall \
    --path ../data/goldman_synthetic_data/ \
    --checkpoint $CHECKPOINT_DIR \
    --output $OUTDIR \
    --chunk-size 2000 --window-overlap 0 --batch-size 64 \
    --beam-size 3 --beam-threshold 0.0 --verbose --overwrite \
    --decoder beam_search --model SACallLoRA --is-synthetic \
    --lora-rank 4 -qk \
    | tee -a logs/beam_search.log

  python -um main.scripts.evaluate_model \
    --reference ../data/goldman_synthetic_data \
    --predictions $OUTDIR \
    --verbose --is-synthetic \
    | tee -a logs/beam_search.log
done