#!/bin/bash
set -e

python segment/data/prepare_data.py \
    --data_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/train_val_data.json \
    --out_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/data \
    --config_path configs/preprocess_pipeline.yaml \
    --train_file_name train_dataset.pt \
    --valid_file_name valid_dataset.pt \
    --fold 1 
