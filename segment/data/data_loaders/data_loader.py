import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from ..data_readers import Kits19Dataset

class Repos(nn.Module):
    def __init__(self, df, fold:int=None, augs=None):
        self.fold = fold
        self.augs = augs
        self.df = df

    def _split_kflod(self, df, fold):
        if fold is not None:
            train_df = df.loc[df["fold"] != fold].reset_index(drop=True)
            valid_df = df.loc[df["fold"] == fold].reset_index(drop=True)

            train_ds = Kits19Dataset(train_df, aug=self.augs, phase="train")
            valid_ds = Kits19Dataset(valid_df, aug=None, phase="val")

        if fold is None:
            train_ds = Kits19Dataset(df, aug=self.augs, phase="train")
            valid_ds = None
        return train_ds, valid_ds

    def _get_repos(self):
        train_ds, valid_ds = self._split_kflod(df=self.df, fold=self.fold)
        return train_ds, valid_ds

    @classmethod
    def get_dloader(cls, df, fold: int = 0, augs=None, verbose=False, **kwargs):
        ds = cls(df=df, fold=fold, augs=augs)
        train_ds, valid_ds = ds._get_repos()
        print("data train:", len(train_ds))
        print("data val:", len(valid_ds)) if valid_ds is not None else None

        train_dl = train_ds.get_loader(**kwargs)
        val_dl = valid_ds.get_loader(**kwargs) if valid_ds is not None else None

        # Illustrate
        idx = np.random.choice(len(train_ds))        
        img_ds, seg_ds = train_ds[idx]
        print("\nimg_ds shape:", img_ds.shape)
        print("seg_ds shape:", seg_ds.shape)
        print("seg_ds value:", np.unique(seg_ds))

        img_dl, seg_dl = next(iter(train_dl))
        print("\nimg_dl shape:", img_dl.shape)
        print("seg_dl shape:", img_dl.shape)
        print("seg_dl value:", np.unique(seg_dl))

        if verbose:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
            ax1.imshow(img[0, img.shape[1] // 2])
            ax2.imshow(seg[0, seg.shape[1] // 2])
            ax3.imshow(seg[1, seg.shape[1] // 2])
            plt.show()

        return train_dl, val_dl