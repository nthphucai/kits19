from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from ..data.augs import aug_maps
from ..utils.utils import get_progress


class Inference:
    def __init__(self, model: nn.Module, data: List[dict], configs: dict):
        self.model = model
        self.data = data
        self.configs = configs

        self.per_device_test_batch_size = 4

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def create(self) -> List[dict]:
        test_dataset = list(
            map(self._get_item, get_progress(self.data, desc="inference "))
        )

        dl_configs = {"batch_size": self.per_device_test_batch_size}
        test_dataloaders = self.get_test_dataloader(test_dataset, **dl_configs)
        minibatch = next(iter(test_dataloaders))
        print("test data shape:", minibatch[1].shape)

        prediction = self._predict(model=self.model, dataloaders=test_dataloaders)
        prediction = self._merge_msk(prediction)
        return prediction

    def get_test_dataloader(self, dataset, **kwargs) -> DataLoader:
        shuffle = kwargs.get("shuffle", False)
        workers = kwargs.get("workers", True)
        pin_memory = kwargs.get("pin_memory", True)
        drop_last = kwargs.get("drop_last", True)
        batch_size = kwargs.get("batch_size", 2)

        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def _merge_msk(self, prediction) -> List[dict]:
        preds = [np.argmax(pred["pred"], axis=0) for pred in prediction]
        case_id = [pred["case_id"] for pred in prediction]
        result = [{"case_id": idc, "pred": pred} for idc, pred in zip(case_id, preds)]
        return result

    def _get_item(self, item):
        case_id = item["case_id"]
        vol_path = item["new_vol_path"]
        vol_arr = np.load(vol_path)
        dropped_vol_arr = aug_maps["crop_and_pad_if_needed"](
            vol_arr, axes=(0, 1, 2), crop_size=self.configs["input_size"]
        )
        vol_tensor = torch.FloatTensor(dropped_vol_arr).unsqueeze(0)
        return case_id, vol_tensor

    @staticmethod
    def _predict(
        model: nn.Module, dataloaders: DataLoader, device="cuda"
    ) -> List[dict]:
        preds_lst = []
        case_id_lst = []

        model.eval()
        model.to(device)
        with torch.no_grad():
            with get_progress(enumerate(dataloaders)) as pbar:
                for _, (case_id, imgs) in enumerate(dataloaders):
                    imgs = imgs.to(device)
                    preds = model(imgs)
                    pbar.update()

                    preds_lst.append(preds.detach().cpu().numpy())
                    case_id_lst.append(case_id)

        preds = np.concatenate(preds_lst)
        case_id = np.concatenate(case_id_lst)
        result = [
            {"case_id": idc, "pred": prediction}
            for idc, prediction in zip(case_id, preds)
        ]
        return result
