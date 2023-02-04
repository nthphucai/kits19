import numpy as np
import pandas as pd
import torch


def dice(
    preds: np.ndarray,
    trues: np.ndarray,
    binary: True,
    threshold: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Returns: Dice score.
    """
    scores = []
    if binary:
        preds = (preds >= threshold).astype(np.float32)

    assert preds.shape == trues.shape
    for b in range(preds.shape[0]):
        pred = preds[b]
        true = trues[b]
        intersection = 2.0 * (true * pred).sum()
        union = true.sum() + pred.sum()
        if true.sum() == 0 and pred.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)

        return np.mean(scores)


class Dice_3D:
    def __init__(self, binary=False, threshold: float = 0.5):
        self.threshold: float = threshold
        self.binary = binary
        self.dsc_class_organ: list = []
        self.dsc_class_tumor: list = []
        self.dsc_scores: list = []

    def init_list(self):
        self.dsc_class_organ = []
        self.dsc_class_tumor = []
        self.dsc_scores = []

    def update(self, preds: np.ndarray, trues: np.ndarray):

        if self.binary:
            final_preds = (preds >= self.threshold).astype(np.float32)
            final_trues = trues

        else:
            preds = np.argmax(preds, axis=1)
            preds_organ = preds.copy()
            preds_organ[preds_organ != 1] = 0
            preds_organ[preds_organ == 1] = 1

            preds_tumor = preds.copy()
            preds_tumor[preds_tumor != 2] = 0
            preds_tumor[preds_tumor == 2] = 1
            final_preds = np.stack([preds_organ, preds_tumor], axis=1)

            trues = np.argmax(trues, axis=1)
            trues_organ = trues.copy()
            trues_organ[trues_organ != 1] = 0
            trues_organ[trues_organ == 1] = 1

            trues_tumor = trues.copy()
            trues_tumor[trues_tumor != 2] = 0
            trues_tumor[trues_tumor == 2] = 1
            final_trues = np.stack([trues_organ, trues_tumor], axis=1)

            assert final_trues.shape == final_preds.shape

        self.update_class(final_preds, final_trues)
        self.update_batch(final_preds, final_trues)

    def update_class(self, preds: np.ndarray, trues: np.ndarray):

        dsc_organ, dsc_tumor = self.dsc_class(preds, trues)

        self.dsc_class_organ.append(dsc_organ)
        self.dsc_class_tumor.append(dsc_tumor)

    def get_dsc_class(self) -> np.ndarray:

        dsc_class_organ = np.mean(self.dsc_class_organ)
        dsc_class_tumor = np.mean(self.dsc_class_tumor)

        return dsc_class_organ, dsc_class_tumor

    def update_batch(self, preds, trues):

        dice = self.dsc_batch(preds, trues)

        self.dsc_scores.append(dice)

    def get_dsc_batch(self) -> np.ndarray:

        dsc_scores = np.mean(self.dsc_scores)

        return dsc_scores

    """
      Params:
          preds: (batch, class, c, w, h)
          trues: (batch, class, c, w, h)
    """

    @staticmethod
    def dsc_batch(preds: np.ndarray, trues: np.ndarray, eps: float = 1e-9):
        """
        Returns: Dice score for data batch.
        """
        scores = []
        for b in range(preds.shape[0]):
            pred = preds[b]
            true = trues[b]
            intersection = 2.0 * (true * pred).sum()
            union = true.sum() + pred.sum()
            if true.sum() == 0 and pred.sum() == 0:
                scores.append(1.0)
            else:
                scores.append((intersection + eps) / union)

        return np.mean(scores)

    @staticmethod
    def dsc_class(preds: np.ndarray, trues: np.ndarray, eps: float = 1e-9):
        """
        Returns: dict with dice scores for data batch and each class.
        """
        classes = np.array(["organ", "tumor"])
        scores = {key: list() for key in classes}

        for b in range(trues.shape[0]):
            for c, class_ in enumerate(classes):
                true = trues[b][c]
                pred = preds[b][c]
                intersection = 2.0 * (true * pred).sum()
                union = true.sum() + pred.sum()

                if true.sum() == 0 and pred.sum() == 0:
                    scores[f"{class_}"].append(1)
                else:
                    scores[f"{class_}"].append((intersection + eps) / union)

        return np.mean(scores["organ"]), np.mean(scores["tumor"])
