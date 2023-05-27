#!/bin/bash
set -e

python segment/training/train.py \
    --model_name_or_path UnetRes_v2 \
    --output_dir output/models/fold_1/best_model_1405_softmax.pt \
    --train_dataset_path output/data/valid_dataset.pt \
    --valid_dataset_path output/data/valid_dataset.pt \
    --config_dir configs/segment_pipeline.yaml \
    --class_weight_path output/class_weight.npy \
    --num_train_epochs 3 \
    --log_dir output/logs/fold_1/logs_1405_softmax.csv \
    --freeze_feature False \
    --do_train True \
    --do_eval False \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --report_to "wandb"
