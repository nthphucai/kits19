import torch

from .lookahead import Lookahead

optimizer_maps = {"adam": torch.optim.Adam, "look_ahead": Lookahead}
