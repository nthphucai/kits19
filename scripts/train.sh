#!/bin/bash
set -e

python segment/training/train.py \
    --output_dir output/models/simple-question/v1.4 \
    --data_path /content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/data/kits19_data.pt \
    --num_epoch 20