#!/bin/bash
set -e

python segment/inference/infer_pipeline.py \
    --model_name_or_path UnetRes_v2 \
    --cache_dir output/models/best_model_2703_softmax.pt \
    --freeze_feature False \
    --num_classes 3 \
    --act_func softmax \
    --config_path configs/preprocess_pipeline.yaml \
    --data_path output/test_data.json \
    --pred_path output/predictions.json \
    --out_path output/predictions \
