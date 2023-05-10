import glob
import os
from abc import ABC, abstractmethod
from itertools import chain
from typing import List

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import DataLoader as load_batch

from segment.utils.file_utils import logger, write_json_file
from segment.utils.utils import get_progress, multiprocess


class BasePreprocess3D(ABC):
    def __init__(
        self,
        data: List[dict],
        vol_path: str = None,
        seg_path: str = None,
    ):
        self.data = data
        self.vol_path = vol_path
        self.seg_path = seg_path

        if (vol_path is not None) and not os.path.exists(vol_path):
            os.makedirs(vol_path)

        if (seg_path is not None) and not os.path.exists(seg_path):
            os.makedirs(seg_path)

    def run(self) -> list:
        data_loader = load_batch(
            dataset=self.data, batch_size=4, collate_fn=self._generate_batch
        )

        temp_lst = []
        for _, data_batch in get_progress(
            enumerate(data_loader), total=len(data_loader)
        ):
            out = multiprocess(
                self.create_one_item, data_batch, workers=4, disable=True
            )
            case_id = [item["case_id"] for item in out]
            cropped_vol = [item["vol"] for item in out]
            cropped_msk = [item["msk"] for item in out]

            cropped_vol_path = [os.path.join(self.vol_path, f"{id_}_imaging.nii.gz") for id_ in case_id]
            cropped_seg_path = [os.path.join(self.seg_path, f"{id_}_segmentation.nii.gz") for id_ in case_id]

            [np.save(vol_path, vol) for vol_path, vol in zip(cropped_vol_path, cropped_vol)]
            [np.save(seg_path, msk) for seg_path, msk in zip(cropped_seg_path, cropped_msk)]

            result = [
                {"case_id": id_, "new_vol_path": vol_path + ".npy", "new_seg_path": seg_path + ".npy"}
                for id_, vol_path, seg_path in zip(
                    case_id, cropped_vol_path, cropped_seg_path
                )
            ]
            temp_lst.append(result)
            del out, result, case_id, cropped_vol, cropped_msk

        flat_out = list(chain(*temp_lst))
        return flat_out

    def to_dict(self):
        cropped_vol_path = glob.glob(f"{self.vol_path}/*")
        cropped_seg_path = [os.path.join(self.seg_path, os.path.basename(path).replace("imaging", "segmentation")) for path in cropped_vol_path]
        case_id = [os.path.basename(file).split("_imaging")[0] for file in cropped_vol_path]

        result = [
                {"case_id": id_, "new_vol_path": vol_path , "new_seg_path": seg_path}
                for id_, vol_path, seg_path in zip(
                    case_id, cropped_vol_path, cropped_seg_path
                )
            ]
        return result        

    def _generate_batch(self, batch) -> dict:
        batch_dict = [
            {
                "case_id": example["case_id"],
                "img_path": example["img_path"],
                "seg_path": example["seg_path"],
            }
            for example in batch
        ]
        return batch_dict

    @staticmethod
    def resample(v, dxyz, new_dxyz, order=1):
        dz, dy, dx = dxyz[:3]
        new_dz, new_dy, new_dx = new_dxyz[:3]

        z, y, x = v.shape

        new_x = np.round(x * dx / new_dx)
        new_y = np.round(y * dy / new_dy)
        new_z = np.round(z * dz / new_dz)

        new_v = zoom(v, (new_z / z, new_y / y, new_x / x), order=order)
        return new_v

    @staticmethod
    def drop_invalid_range(volume, label=None):
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return (
                volume[min_z:max_z, min_h:max_h, min_w:max_w],
                label[min_z:max_z, min_h:max_h, min_w:max_w],
            )
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]

    @abstractmethod
    def create_one_item(self, data):
        pass
