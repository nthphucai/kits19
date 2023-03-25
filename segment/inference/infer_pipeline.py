import os
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import numpy as np
import nibabel as nib

from segment.utils.file_utils import load_json_file, write_json_file, logger, read_yaml_file
from segment.models.segment import get_model
from segment.models import model_maps
from segment.inference.infer import Inference
from segment.utils.hf_argparser import HfArgumentParser
from segment.data.postprocess import Postprocess3D

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from ..."},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models"},
    )

    freeze_feature: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze_feature"},
    )

    num_classes: Optional[int] = field(
        default=3,
        metadata={"help": "The number of classes to be classified"},

    )

    act_func: Optional[str] = field(
        default="softmax", metadata={"help": "activate function"}
    )

    config_path: Optional[str] = field(
        default="configs/preprocess_pipeline.yaml.yaml",
        metadata={"help": "Path to config file"},
    )


@dataclass
class DataInferenceArguments:
    data_path: Optional[str] = field(
        default="output/test_data.json", 
        metadata={"help": "Path to data"}
    )

    out_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save prediction"}
    )


def inference(
    data_path:str, 
    config_path:str,
    out_path:str=None,
    model_name_or_path:str="UnetRes_v2",
    pretrained_path:str="output/models/best_model_1803_softmax.pt",
    freeze_feature: bool=False,  
    num_classes:int=3,
    act_func:str="softmax"
):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    
    configs = read_yaml_file(config_path)
    data = load_json_file(data_path)["data"]
    model = get_model(
        model=model_maps[model_name_or_path],
        pretrained_path=pretrained_path,
        freeze_feature=freeze_feature,
        num_classes=num_classes,
        act=act_func,
    ) 

    inference = Inference(model=model, data=data, configs=configs["create_dataset"])
    prediction = inference.create()
    
    case_id = [item["case_id"].split("_")[1] for item in prediction]
    npy_pred_path = [out_path + f"/prediction_{idc}.nii.gz.npy" for idc in case_id]
    nii_pred_path = [f"predictions/prediction_{idc}.nii.gz" for idc in case_id]

    npy_msk_pred = [item["pred"] for item in prediction]
    
    assert len(case_id) == len(npy_msk_pred), f"len{len(case_id)} and len{len(npy_msk_pred)}"

    [np.save(path, seg) for path, seg in zip(npy_pred_path, npy_msk_pred)]
    [item.update({"npy_seg_path": npy_pred_path[idc]}) for idc, item in enumerate(data)]

    postprocess = Postprocess3D(configs=configs["preprocess"], prediction=data)
    nii_msk_pred, case_id = postprocess.create()

    [nib.save(nii_pred, path) for nii_pred, path in zip(nii_msk_pred, nii_pred_path)]
    [item.update({"nii_seg_path": nii_pred_path[idc]}) for idc, item in enumerate(data)]

    write_json_file({"data": data}, "output/test_data_update.json")
    logger.info(f"The number of test dataset is {len(nii_msk_pred)}")
    logger.info(f"Saved test dataset at {out_path}")

def main():
    parser = HfArgumentParser((ModelArguments, DataInferenceArguments))
    model_args, data_infer_args = parser.parse_args_into_dataclasses()
    
    inference(
        data_path=data_infer_args.data_path, 
        out_path=data_infer_args.out_path,
        model_name_or_path=model_args.model_name_or_path,
        config_path=model_args.config_path,
        pretrained_path=model_args.cache_dir,
        freeze_feature=model_args.freeze_feature,  
        num_classes=model_args.num_classes,
        act_func=model_args.act_func
    )


if __name__ == "__main__":
    main()
