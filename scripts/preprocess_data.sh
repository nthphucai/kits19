#!/bin/bash
set -e

python segment/data/preprocess_data.py \
    --data_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/data/train_ds.csv \
    --train_vol_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/vol \
    --train_seg_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/seg \
    --save_file_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/train_val_data.json \
    --config_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/configs/preprocess_pipeline.yaml \
    --split_kfold 10 \
