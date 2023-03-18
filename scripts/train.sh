#!/bin/bash
set -e

python segment/training/train.py \
    --model_name_or_path UnetRes_v2 \
    --output_dir output/models \
    --config_dir configs/segment_pipeline.yaml \
    --num_classes 2 \
    --act_func sigmoid \
    --data_path output/data/train_dataset.pt \
    --class_weight_path output/class_weight.npy \
    