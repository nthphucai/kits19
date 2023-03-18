#!/bin/bash
set -e

python segment/training/train.py \
    --model_name_or_path UnetRes_v2 \
    --output_dir output/models/best_model_1803.pt \
    --config_dir configs/segment_pipeline.yaml \
    --num_classes 2 \
    --act_func sigmoid \
    --train_dataset_path output/data/train_dataset.pt \
    --valid_dataset_path output/data/valid_dataset.pt \
    --class_weight_path output/class_weight.npy \
    --num_train_epochs 3 \
    --log_dir output/logs_1803.csv \
