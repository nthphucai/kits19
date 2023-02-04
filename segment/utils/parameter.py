import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

lower = -79
upper = 304

lower_bound = -79
upper_bound = 304

expand_slice = 20

target_spacing = [3, 1.5, 1.5]

# Direction

output_path = "/Users/HPhuc/Practice/8. Kits2019/Source/outputs/"

csv_path = Path(output_path, "csv/")

log_path = Path(output_path, "logs/")

config_path = Path(output_path, "configs/")

cpoints_path = Path(output_path, "checkpoints/")

# Parameters
lower_bound = -79

upper_bound = 304

mean = 101

std = 76.9

vol_min = -122.0

vol_max = 470.0

size = 48

expand_slice = 20

bz = 1

workers = 2

pin_memory = True

expand_slice = 20

learning_rate = 1e-4

learning_rate_decay = [500, 750]

drop_rate = 0.5

num_classes = 3
