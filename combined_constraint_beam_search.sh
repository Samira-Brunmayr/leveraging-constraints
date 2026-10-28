#!/bin/bash

mkdir -p logs


OUTDIR="../experiments/finetune/whole_model_combined"
python -um main.scripts.basecall \
  --path ../data/goldman_synthetic_data/ \
  --checkpoint ../experiments/finetune/models/5000_bacterial_benchmark_finetuned_45000/checkpoints/checkpoint_45000.pt \
  --output $OUTDIR \
  --chunk-size 2000 --window-overlap 0 --batch-size 64 \
  --beam-size 3 --beam-threshold 0.0 --verbose \
  --decoder beam_search --model SACallLoRA --is-synthetic \
  --max-repeats 1 --discard-kl 1 \
  --desired-length 117 --length-lambda 5 \
  | tee -a logs/beam_search.log

python -um main.scripts.evaluate_model \
  --reference ../data/goldman_synthetic_data \
  --predictions $OUTDIR \
  --verbose --is-synthetic \
  | tee -a logs/beam_search.log

OUTDIR="../experiments/finetune/decoder_combined"
python -um main.scripts.basecall \
  --path ../data/goldman_synthetic_data/ \
  --checkpoint ../experiments/finetune/models/5000_images_model_decoder_finetune_45000/checkpoints/checkpoint_45000.pt \
  --output $OUTDIR \
  --chunk-size 2000 --window-overlap 0 --batch-size 64 \
  --beam-size 3 --beam-threshold 0.0 --verbose \
  --decoder beam_search --model SACallLoRA --is-synthetic \
  --max-repeats 1 --discard-kl 1 \
  --desired-length 117 --length-lambda 5 \
  | tee -a logs/beam_search.log

python -um main.scripts.evaluate_model \
  --reference ../data/goldman_synthetic_data \
  --predictions $OUTDIR \
  --verbose --is-synthetic \
  | tee -a logs/beam_search.log

for LAMBDA in 1 2 4 8 16; do
  CHECKPOINT_DIR="../experiments/finetune/models/5000_images_model_finetune_rank_${LAMBDA}_45000/checkpoints/checkpoint_45000.pt"
  OUTDIR="../experiments/finetune/lora_${LAMBDA}_combined/"
  python -um main.scripts.basecall \
    --path ../data/goldman_synthetic_data/ \
    --checkpoint $CHECKPOINT_DIR \
    --output $OUTDIR \
    --chunk-size 2000 --window-overlap 0 --batch-size 64 \
    --beam-size 3 --beam-threshold 0.0 --verbose \
    --decoder beam_search --model SACallLoRA --is-synthetic \
    --max-repeats 1 --discard-kl 1 \
    --desired-length 117 --length-lambda 5 \
    --lora-rank $LAMBDA -qk \
    | tee -a logs/beam_search.log

  python -um main.scripts.evaluate_model \
    --reference ../data/goldman_synthetic_data \
    --predictions $OUTDIR \
    --verbose --is-synthetic \
    | tee -a logs/beam_search.log
done