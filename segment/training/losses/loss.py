import numpy as np
import pandas as pd
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, smooth=1e-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, preds, trues):
        assert (
            preds.shape == trues.shape
        ), f"predition shape {preds.shape} is not the same as label shape {trues.shape}"

        trues = trues.float()
        bce = self.bce(preds, trues)
        bce = bce.clip(self.smooth, 1.0 - self.smooth)
        pt = torch.exp(-bce)
        focal_bce = bce * (1 - pt) ** self.gamma

        return focal_bce.mean()


class DiceLoss(nn.Module):
    """Calculate dice loss."""

    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, trues: torch.Tensor) -> torch.Tensor:
        num = trues.size(0)
        preds = preds.view(num, -1)
        trues = trues.view(num, -1)

        assert preds.shape == trues.shape
        intersection = 2.0 * (preds * trues).sum(axis=-1)
        union = preds.sum(axis=-1) + trues.sum(axis=-1)
        dice = (intersection + self.smooth) / union

        return (1.0 - dice).mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2, smooth: float = 1e-6, weight=0.1):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(gamma=gamma)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.weight = weight

    def forward(self, preds, trues):
        return (
            self.weight * self.focal_loss(preds, trues) + self.dice_loss(preds, trues)
        ) / (self.weight + 1)


class Dice_BCE(nn.Module):
    def __init__(self, smooth=1e-8, label_smoothing=0.01, class_weight=None):
        super().__init__()
        self.eps = smooth
        self.lsm = label_smoothing
        self.class_weight = class_weight

        if self.class_weight is not None:
            self.class_weight = torch.tensor(
                class_weight, dtype=torch.float, device=device
            )

    def forward(self, preds, trues):
        p = (preds * trues).sum(dim=[-1, -2, -3])
        s = (preds + trues).sum(dim=[-1, -2, -3])
        dice = (2 * p + 1) / (s + 1)

        ln0 = (1 - preds + 1e-6).log()
        ln1 = (preds + 1e-6).log()

        if self.lsm is not None:
            l1 = (1 - self.lsm) * trues * ln1 + (self.lsm / 2) * ln1
            l2 = (1 - self.lsm) * (1 - trues) * ln0 + (self.lsm / 2) * ln0

        else:
            l1 = (1 - trues) * ln0
            l2 = trues * ln1

        ce = l1 + l2

        if self.class_weight is not None:
            ce = -ce * self.class_weight
        else:
            ce = -ce

        return (1 - dice).mean() + ce.mean()


class BCE(nn.Module):
    def __init__(self, label_smoothing=0.01):
        super().__init__()
        self.lsm = label_smoothing

    def forward(self, preds, trues):
        ln0 = (1 - preds + 1e-6).log()
        ln1 = (preds + 1e-6).log()

        if self.lsm is not None:
            l1 = (1 - self.lsm) * trues * ln1 + (self.lsm / 2) * ln1
            l2 = (1 - self.lsm) * (1 - trues) * ln0 + (self.lsm / 2) * ln0

        else:
            l1 = (1 - trues) * ln0
            l2 = trues * ln1

        loss = l1 + l2

        return -loss.mean().item()


class WBCE(nn.Module):
    def __init__(self, weights_path=None, smooth=None):
        super().__init__()

        if weights_path is None:
            weights = np.array([[1, 1]])
            print("using default weight")
            print(pd.DataFrame(weights, index=["default"]))
        elif ".csv" in weights_path:
            weights = pd.read_csv(weights_path, index_col=0)
            print(weights)
            weights = weights.values
        elif ".npy" in weights_path:
            weights = np.load(weights_path)
            print(weights.shape)
        else:
            raise NotImplementedError("only support csv and numpy extension")

        self.weights = torch.tensor(weights, dtype=torch.float, device=device)
        self.smooth = smooth

    def forward(self, preds, trues):
        ln0 = (1 - preds + 1e-7).log()
        ln1 = (preds + 1e-7).log()

        weights = self.weights
        if self.smooth is not None:
            sm = torch.ones_like(preds).uniform_(1 - self.smooth, 1)
            ln0 = weights[..., 0] * (1 - trues) * (sm * ln0 + (1 - sm) * ln1)
            ln1 = weights[..., 1] * trues * (sm * ln1 + (1 - sm) * ln0)
        else:
            ln0 = weights[..., 0] * (1 - trues) * ln0
            ln1 = weights[..., 1] * trues * ln1

        ln = ln0 + ln1
        return -ln.mean()

    def extra_repr(self):
        return f"weights_shape={self.weights.shape}, smooth={self.smooth}, device={self.weights.device}"

    @staticmethod
    def get_class_balance_weight(counts, anchor=0):
        """
        calculate class balance weight from counts with anchor
        :param counts: class counts, shape=(n_class, 2)
        :param anchor: make anchor class weight = 1 and keep the aspect ratio of other weight
        :return: weights for cross entropy loss
        """
        total = counts.values[0, 0] + counts.values[0, 1]
        beta = 1 - 1 / total

        weights = (1 - beta) / (1 - beta**counts)
        normalized_weights = weights / weights.values[:, anchor, np.newaxis]

        return normalized_weights
