from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import numpy as np

from segment.utils.file_utils import load_json_file, write_json_file, logger
from segment.models.segment import get_model
from segment.models import model_maps
from segment.inference.infer_pipeline import Inference
from segment.utils.hf_argparser import HfArgumentParser


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

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
    out_path:str=None,
    model_name_or_path:str="UnetRes_v2",
    pretrained_path:str="output/models/best_model_1803_softmax.pt",
    freeze_feature: bool=False,  
    num_classes:int=3,
    act_func:str="softmax"
):

    data = load_json_file(data_path)["data"]
    model = get_model(
        model=model_maps[model_name_or_path],
        pretrained_path=pretrained_path,
        freeze_feature=freeze_feature,
        num_classes=num_classes,
        act=act_func,
    ) 
    inference = Inference(model=model, data=data)
    prediction = inference.create()

    case_id = [item["case_id"] for item in prediction]
    msk_pred = [item["pred"] for item in prediction]

    pred_path = [out_path + f"/{idc}_imaging.nii.gz" for idc in case_id]
    [np.save(path, seg) for path, seg in zip(pred_path, msk_pred)]

    logger.info(f"The number of test dataset is {len(msk_pred)}")
    logger.info(f"Saved test dataset at {out_path}")

def main():
    parser = HfArgumentParser((ModelArguments, DataInferenceArguments))
    model_args, data_infer_args = parser.parse_args_into_dataclasses()
    
    inference(
        data_path=data_infer_args.data_path, 
        out_path=data_infer_args.out_path,
        model_name_or_path=model_args.model_name_or_path,
        pretrained_path=model_args.cache_dir,
        freeze_feature=model_args.freeze_feature,  
        num_classes=model_args.num_classes,
        act_func=model_args.act_func
    )


if __name__ == "__main__":
    main()
