from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ..augs import aug_maps
from ..normalization import normalize
from .standard_class import StandardDataset


class DatasetReader(StandardDataset):
    def __init__(self, 
        df=None, 
        augs=None,
        phase:Optional[str]="train",
        input_size:Union[tuple,list]=(80, 80, 160),
        expand_slice_z:int=20,
        expand_slice_y:int=20,
        expand_slice_x:int=20,
        crop_type:str="random",
        use_bground:bool=False,
        z_range:Union[tuple, list]=None,
        y_range:Union[tuple, list]=None,
        x_range:Union[tuple, list]=None
    ):
        super().__init__()
        self.df = df
        self.augs = augs
        self.phase = phase
        self.patch_size = input_size
        self.crop_type = crop_type
        self.use_bground = use_bground

        self.expand_slice_z = expand_slice_z
        self.expand_slice_y = expand_slice_y
        self.expand_slice_x = expand_slice_x

        self.z_range = z_range
        self.y_range = y_range
        self.x_range = x_range

    def get_len(self):
        return self.df.shape[0]

    def get_item(self, idx):
        row = self.df.iloc[idx]
        vol_path = row["new_vol_path"]
        seg_path = row["new_seg_path"]
        vol_arr = np.load(vol_path)
        seg_arr = np.load(seg_path)

        dropped_vol_arr, dropped_seg_arr = self._drop_zero_slice(vol_arr, seg_arr)
        dropped_vol_arr, dropped_seg_arr = self._crop_or_pad(dropped_vol_arr, dropped_seg_arr, self.crop_type, vol_arr, seg_arr)

        dropped_seg_arr = self._stack_mask(dropped_seg_arr, use_bground=self.use_bground)
        # print(self.locate_nonzero_slice(torch.FloatTensor(dropped_seg_arr[0]), t=1))

        vol_tensor = torch.FloatTensor(dropped_vol_arr).unsqueeze(0)
        seg_tensor = torch.FloatTensor(dropped_seg_arr)
        
        if self.augs is not None:
            auged = self.augs({"image": vol_tensor, "mask": seg_tensor})
            vol_tensor = auged["image"]
            seg_tensor = (auged["mask"] > 0.5).int()

        return vol_tensor, seg_tensor

    def _stack_mask(self, seg_arr, use_bground: False) -> np.array:
        """This function aims to stack organ mask and tumor mask into one single mask.

        Args:
            seg_arr (np.ndarray): segment array
            use_bground (False): whether to use background as one class to classify,
            in this case, there are 3 classes (background, organ, tumor)

        Returns:
            numpy.array: the segmented mask.
        """
        pos_seg = seg_arr.copy()

        bground_arr = seg_arr.copy()
        bground_arr[pos_seg != 0] = 0
        bground_arr[pos_seg == 0] = 1

        organ_arr = seg_arr.copy()
        organ_arr[pos_seg != 1] = 0
        organ_arr[pos_seg == 1] = 1

        tumor_arr = seg_arr.copy()
        tumor_arr[pos_seg != 2] = 0
        tumor_arr[pos_seg == 2] = 1

        if use_bground:
            return np.stack([bground_arr, organ_arr, tumor_arr])
        else:
            return np.stack([organ_arr, tumor_arr])

    def _drop_zero_slice(self, vol_arr: np.array, seg_arr: Optional[np.array]=None):
        vol_arr = vol_arr[
            int(self.z_range[0] - self.expand_slice_z) : int(self.z_range[1] + self.expand_slice_z), 
            int(self.y_range[0] - self.expand_slice_y) : int(self.y_range[1] + self.expand_slice_y), 
            int(self.x_range[0] - self.expand_slice_x) : int(self.x_range[1] + self.expand_slice_x)
        ]

        if seg_arr is not None:
          seg_arr = seg_arr[
              int(self.z_range[0] - self.expand_slice_z) : int(self.z_range[1] + self.expand_slice_z), 
              int(self.y_range[0] - self.expand_slice_y) : int(self.y_range[1] + self.expand_slice_y), 
              int(self.x_range[0] - self.expand_slice_x) : int(self.x_range[1] + self.expand_slice_x)
          ]
          return vol_arr, seg_arr
        else:
          return vol_arr

    def _crop_or_pad(self, 
          vol_arr: np.array, 
          seg_arr: np.array, 
          crop_type: Union[list, tuple, int], 
          vol_arr_org: Union[np.array], 
          seg_arr_org: Union[np.array]
      ):
        if crop_type == "random":
            vol_arr, seg_arr = aug_maps["random_crop_3D"](img=vol_arr, label=seg_arr, crop_size=self.patch_size)
        
        elif crop_type == "random_center_crop":
          vol_arr, seg_arr = aug_maps["random_center_crop"](vol_arr, seg_arr)

        elif crop_type == "center":
            vol_arr = aug_maps["crop_and_pad_if_needed"](vol_arr, axes=(0,1,2), crop_size=self.patch_size)
            seg_arr = aug_maps["crop_and_pad_if_needed"](seg_arr, axes=(0,1,2), crop_size=self.patch_size)

        out = self.locate_nonzero_slice(torch.FloatTensor(seg_arr))
        if out == "empty sequence":
            return self._crop_or_pad(vol_arr_org, seg_arr_org, self.crop_type, vol_arr_org, seg_arr_org)
        del out
        return vol_arr, seg_arr

    @staticmethod
    def locate_nonzero_slice(volume, t:int=0, return_rate=True):
        assert len(volume.size()) == 3, "The input size must be (z, h, w)"
        assert isinstance(volume, torch.Tensor)

        nonzero_z = np.where(volume.sum(dim=(1,2)) > t)[0]
        nonzero_y = np.where(volume.sum(dim=(0,2)) > t)[0]   
        nonzero_x = np.where(volume.sum(dim=(0,1)) > t)[0]
        try:
          if not return_rate:
              return {
                  "nonzero_z_min": min(nonzero_z), "nonzero_z_max": max(nonzero_z),
                  "nonzero_y_min": min(nonzero_y), "nonzero_y_max": max(nonzero_y),
                  "nonzero_x_min": min(nonzero_x), "nonzero_x_max": max(nonzero_x),
              }
          else:
              return {
                  "rate_z_axis": round((max(nonzero_z) - min(nonzero_z)) / volume.shape[0], 2),
                  "rate_y_axis": round((max(nonzero_y) - min(nonzero_y)) / volume.shape[1], 2),
                  "rate_x_axis": round((max(nonzero_x) - min(nonzero_x)) / volume.shape[2], 2),
              }
        except:
          return "empty sequence"

        

  

        
