from typing import List

import torch
import torch.nn as nn 
import numpy as np
from torch.utils.data import DataLoader

from segment.data.augs import aug_maps
from segment.utils.utils import get_progress, multiprocess
from segment.data.preprocess_data import preprocess_volume
from segment.data.data_loaders.processor import DataProcessor

class Inference:
    def __init__(self, model: nn.Module, data: List[dict]):
        self.model = model
        self.data = data

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def create(self) -> List[dict]:
        test_dataset = list(map(self._get_item, get_progress(self.data)))
        processor = DataProcessor(batch_size=4, workers=2)
        test_dataloaders = processor.get_dloader(test_dataset) 
        
        prediction = self._predict(model=self.model, dataloaders=test_dataloaders)
        prediction = self._merge_msk(prediction)
        return prediction

    def _merge_msk(self, prediction) -> List[dict]:
        preds = [np.argmax(pred["pred"], axis=0) for pred in prediction]
        case_id = [pred["case_id"] for pred in prediction]
        result = [{"case_id": idc, "pred": pred} for idc, pred in zip(case_id, preds)]
        return result

    def _get_item(self, item):
        case_id = item["case_id"]
        vol_path = item["new_vol_path"]
        vol_arr = np.load(vol_path)
        dropped_vol_arr = aug_maps["crop_and_pad_if_needed"](vol_arr, axes=(0,1,2), crop_size=[80, 80, 160])
        vol_tensor = torch.FloatTensor(dropped_vol_arr).unsqueeze(0)
        return case_id, vol_tensor

    @staticmethod
    def _predict(model: nn.Module, dataloaders:DataLoader, device="cuda") -> List[dict]:
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
        result = [{"case_id": idc, "pred": prediction} for idc, prediction in zip(case_id, preds)]
        return result
