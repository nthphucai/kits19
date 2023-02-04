import os

import nibabel as nib
import numpy as np
import pandas as pd

"""create dataframe
"""
# data_path = '/content/drive/MyDrive/Seg3D/KiTS2019/data/'
# df = pd.DataFrame({ 'case_id' : ['case_' + f'{i:05}' for i in range(20)],
#                     'img_path': [f'{data_path}' + 'case_' + f'{i:05}' + '/imaging.nii.gz' for i in range(20)],
#                     'seg_path': [f'{data_path}' + 'case_' + f'{i:05}' + '/segmentation.nii.gz' for i in range(20)]
#                    })


def df_image_mask_path(root):

    dataset = []
    remnants = []
    arr_foldername = []

    for folder_name in os.listdir(root):
        arr_foldername.append(folder_name)

    for folder_name in sorted(arr_foldername, key=lambda x: str(x.split("_")[-1])):
        arr = []
        nimg = []
        nmask = []
        try:
            for file_name in os.listdir(os.path.join(root, folder_name)):
                found = False
                if "segmentation" in file_name:
                    for name in nimg:
                        nimg.remove(name)
                        found = True
                        break

                    if found:
                        mask = (
                            nib.load(
                                os.path.join(root, folder_name, file_name)
                            ).get_fdata()
                        ).sum()
                        containing_mask = 0

                        if mask > 0:
                            containing_mask = 1
                        dataset.append(
                            [
                                folder_name,
                                os.path.join(root, folder_name, name),
                                os.path.join(root, folder_name, file_name),
                                containing_mask,
                            ]
                        )

                    nmask.append(file_name)

                else:
                    for name in nmask:
                        nmask.remove(name)
                        found = True
                        break

                    if found:
                        mask = (
                            nib.load(os.path.join(root, folder_name, name)).get_fdata()
                        ).sum()
                        containing_mask = 0
                        if mask > 0:
                            containing_mask = 1
                        dataset.append(
                            [
                                folder_name,
                                os.path.join(root, folder_name, file_name),
                                os.path.join(root, folder_name, name),
                                containing_mask,
                            ]
                        )
                    nimg.append(file_name)

        except Exception as e:
            pass

    return pd.DataFrame(dataset, columns=["case_id", "img_path", "seg_path", "mask"])
