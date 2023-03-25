#!/bin/bash
set -e

python segment/training/train.py \
    --model_name_or_path UnetRes_v2 \
    --output_dir output/models/best_model_2403_softmax.pt \
    --cache_dir="output/models/best_model_1903_softmax.pt" \
    --train_dataset_path output/data/train_dataset.pt \
    --valid_dataset_path output/data/valid_dataset.pt \
    --config_dir configs/segment_pipeline.yaml \
    --class_weight_path output/class_weight.npy \
    --num_train_epochs 5 \
    --log_dir output/logs_2403.csv \
    --freeze_feature False \
    --do_train True \
    --do_eval True 
