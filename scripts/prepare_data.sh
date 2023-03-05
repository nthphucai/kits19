#!/bin/bash
set -e

python segment/data/prepare_data.py \
    --data_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/train_val_data.json \
    --out_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/data/kits19_data.pt \
    --fold 1 
