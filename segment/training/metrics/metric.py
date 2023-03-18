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

    def update_one_batch(self, preds: np.ndarray, trues: np.ndarray):
        """
        :param
        :preds: b, c, d, h, w
        :trues: b, c, d, h, w
        """
        if self.binary:
            final_preds = (preds >= self.threshold).astype(np.float32)
            final_trues = (trues >= self.threshold).astype(np.float32)

        else:
            preds_max = np.argmax(preds, axis=1)
            preds_bground = preds_max.copy()
            preds_bground[preds_max != 0] = 0
            preds_bground[preds_max == 0] = 1

            preds_organ = preds_max.copy()
            preds_organ[preds_max != 1] = 0
            preds_organ[preds_max == 1] = 1

            preds_tumor = preds_max.copy()
            preds_tumor[preds_max != 2] = 0
            preds_tumor[preds_max == 2] = 1
            final_preds = np.stack([preds_organ, preds_tumor], axis=1)

            trues_max = np.argmax(trues, axis=1)
            trues_bground = trues_max.copy()
            trues_bground[trues_max != 0] = 0
            trues_bground[trues_max == 0] = 1

            trues_organ = trues_max.copy()
            trues_organ[trues_max != 1] = 0
            trues_organ[trues_max == 1] = 1

            trues_tumor = trues_max.copy()
            trues_tumor[trues_max != 2] = 0
            trues_tumor[trues_max == 2] = 1
            final_trues = np.stack([trues_organ, trues_tumor], axis=1)

            assert final_trues.shape == final_preds.shape

        self.update_class(final_preds, final_trues)
        self.update_batch(final_preds, final_trues)

    def update_class(self, preds: np.ndarray, trues: np.ndarray):
        dsc_organ, dsc_tumor = self.dsc_class_one_batch(preds, trues)

        self.dsc_class_organ.append(dsc_organ)
        self.dsc_class_tumor.append(dsc_tumor)

    @staticmethod
    def dsc_class_one_batch(preds: np.ndarray, trues: np.ndarray, eps: float = 1e-9):
        """
        Params:
          preds: (batch, class, c, w, h)
          trues: (batch, class, c, w, h)
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

    def update_batch(self, preds, trues):
        dice = self.dsc_one_batch(preds, trues)

        self.dsc_scores.append(dice)

    @staticmethod
    def dsc_one_batch(preds: np.ndarray, trues: np.ndarray, eps: float = 1e-9):
        """
        Params:
        preds: (batch, class, c, w, h)
        trues: (batch, class, c, w, h)
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

    def get_dsc_class_one_epoch(self) -> np.ndarray:
        dsc_class_organ = np.mean(self.dsc_class_organ)
        dsc_class_tumor = np.mean(self.dsc_class_tumor)
        return dsc_class_organ, dsc_class_tumor

    def get_total_dsc_one_epoch(self) -> np.ndarray:
        dsc_scores = np.mean(self.dsc_scores)
        return dsc_scores

    def get_dsc_one_epoch(self) -> np.ndarray:
        dsc_class_organ = np.mean(self.dsc_class_organ)
        dsc_class_tumor = np.mean(self.dsc_class_tumor)
        dsc_total_scores = np.mean(self.dsc_scores)
        self.init_list()
        return dsc_total_scores, dsc_class_organ, dsc_class_tumor
