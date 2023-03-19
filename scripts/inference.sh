#!/bin/bash
set -e

python segment/inference/infer.py \
    --model_name_or_path UnetRes_v2 \
    --cache_dir output/models/best_model_1903_softmax.pt \
    --freeze_feature False \
    --num_classes 3 \
    --act_func softmax \
    --data_path output/test_data.json \
    --out_path output/preds \

