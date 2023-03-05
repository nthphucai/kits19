from dataclasses import dataclass, field
from typing import Optional

import torch
import pandas as pd
import numpy as np

from segment.models import model_maps
from segment.models.segment import get_model
from segment.training.trainer.config_trainer import ConfigTrainer
from segment.utils.file_utils import logger, read_yaml_file
from segment.utils.hf_argparser import HfArgumentParser
import segment.utils.parameter as para


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from ..."},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models"},
    )

    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models"},
    )

    class_weight: Optional[str] = field(
        default=None,
        metadata={"help": "Path to store class_weight"},
    )


@dataclass
class DataTrainingArguments:
    data_path: Optional[str] = field(
        default="data/train_ds.csv", metadata={"help": "Path to data"}
    )


@dataclass
class TrainingArguments:
    num_epochs: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, _ = parser.parse_args_into_dataclasses()
    
    class_weight = np.array([0.00673965, 0.27936378, 1.0])
    class_weight = np.array([0.27936378, 1.0]) if para.num_classes == 2 else class_weight

    class_weight = class_weight.reshape(1, para.num_classes, *[1] * 3)
    np.save("/content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/class_weight.npy", class_weight)
    logger.info("Saved class weight at /content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/class_weight.npy")

    model = get_model(
        model=model_maps["UnetRes_v2"],
        pretrained_path=model_args.model_path,
        freeze_feature=False,
        num_classes=para.num_classes,
        act="softmax",
    )

    config = read_yaml_file("configs/segment_pipeline.yaml")["segment_kits"]
    data = torch.load(data_args.data_path)

    trainer = ConfigTrainer(data=data, model=model, config=config)
    trainer.train()


if __name__ == "__main__":
    main()
