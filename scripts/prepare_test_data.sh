#!/bin/bash
set -e

python segment/inference/prepare_test_data.py \
    --data_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/data/ \
    --config_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/configs/preprocess_pipeline.yaml \
    --vol_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/test_data/vol \
    --out_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/test_data.json \
