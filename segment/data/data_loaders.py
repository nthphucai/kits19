import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segment.utils.parameter as para
from segment.data.augs import augs
from segment.data.data_readers import Kits19Dataset

# from repos.training.data.get_ds_1 import Kits19Dataset

"""
get repos: get data from repos
"""


class Repos(nn.Module):
    def __init__(self, df, fold=None, augs=None):
        self.fold = fold
        self.augs = augs
        self.df = df

    def split_kflod(self, df, fold):
        if fold is not None:
            train_df = df.loc[df["fold"] != fold].reset_index(drop=True)
            val_df = df.loc[df["fold"] == fold].reset_index(drop=True)

            train_ds = Kits19Dataset(train_df, aug=self.augs, phase="train")
            val_ds = Kits19Dataset(val_df, aug=None, phase="val")

        if fold is None:
            train_ds = Kits19Dataset(df, aug=self.augs, phase="train")
            val_ds = None
        return train_ds, val_ds

    def get_repos(self):
        train_ds, val_ds = self.split_kflod(df=self.df, fold=self.fold)
        return train_ds, val_ds


"""
get loader: load data from repos
"""


def get_dloader(df, fold: int = 0, augs=augs, **kwargs):
    ds = Repos(df, fold=fold, augs=augs)
    train_ds, val_ds = ds.get_repos()
    print("data train:", len(train_ds))
    print("data val:", len(val_ds)) if val_ds is not None else None

    train_dl = train_ds.get_loader(**kwargs)
    val_dl = val_ds.get_loader(**kwargs) if val_ds is not None else None

    # Illustrate
    idx = np.random.choice(len(train_ds))
    img, seg = train_ds[idx]
    print("img shape:", img.shape)
    print("seg shape:", seg.shape)
    print("seg value:", np.unique(seg))
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,8))
    # ax1.imshow(img[0, img.shape[1]//2])
    # ax2.imshow(seg[0, seg.shape[1]//2])
    # ax3.imshow(seg[1, seg.shape[1]//2])

    return train_dl, val_dl
