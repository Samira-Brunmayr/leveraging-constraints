#!/bin/bash

# Define list of input directories here
INPUT_DIRS=(
  "../experiments/finetune/5000_bacterial_benchmark_finetuned_45000"
  "../experiments/finetune/whole_model_combined"
  "../experiments/finetune/5000_images_model_decoder_finetune_45000"
  "../experiments/finetune/decoder_combined"
  "../experiments/finetune/5000_images_model_finetune_rank_1_45000"
  "../experiments/finetune/lora_1_combined"
  "../experiments/finetune/5000_images_model_finetune_rank_2_45000"
  "../experiments/finetune/lora_2_combined"
  "../experiments/finetune/5000_images_model_finetune_rank_4_45000"
  "../experiments/finetune/lora_4_combined"
  "../experiments/finetune/5000_images_model_finetune_rank_8_45000"
  "../experiments/finetune/lora_8_combined"
  "../experiments/finetune/5000_images_model_finetune_rank_16_45000"
  "../experiments/finetune/lora_16_combined"
)

GOLDMAN_DIR=../data/goldman_synthetic_data

for INPUT_DIR in "${INPUT_DIRS[@]}"; do
  echo "Running on $INPUT_DIR"
  python -m main.scripts.payload_accuracy_synthetic \
    --input-dir "$INPUT_DIR" \
    --goldman-dir "$GOLDMAN_DIR"
done