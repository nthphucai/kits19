#!/bin/bash
set -e

python segment/data/preprocess_data.py \
    --data_path data/train_ds.csv \
    --train_vol_path output/vol \
    --train_seg_path output/seg \
    --save_file_path output/train_val_data.json \
    --config_path configs/preprocess_pipeline.yaml \
    --split_kfold 4 \
