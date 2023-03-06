import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

mean = 101

std = 76.9


bz = 1

workers = 2

pin_memory = True

learning_rate = 1e-4

learning_rate_decay = [500, 750]

drop_rate = 0.5

num_classes = 2
