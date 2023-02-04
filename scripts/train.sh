#!/bin/bash
set -e

python segment/training/train.py \
    --output_dir output/models/simple-question/v1.4 \
    --data_path data/train_ds.csv \
    --num_epoch 200