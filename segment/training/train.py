from dataclasses import dataclass, field
from typing import Optional

import torch
import pandas as pd
import numpy as np

from segment.models import model_maps
from segment.models.segment import get_model
from segment.data.data_loaders.processor import DataProcessor
from segment.training.trainer.config_trainer import ConfigTrainer
from segment.utils.file_utils import logger, read_yaml_file
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

    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models"},
    )

    config_dir: Optional[str] = field(
        default="configs/segment_pipeline.yaml",
        metadata={"help": "Path to config file"},
    )

    num_classes: Optional[int] = field(
      default=2,
      metadata={"help": "The number classes to classify"},
    )

    freeze_feature: Optional[bool] = field(
      default=False,
      metadata={"help": "Whether to freeze_feature"},
    )

    act_func: Optional[str] = field(
      default="softmax",
      metadata={"help": "activate function"}
    )


@dataclass
class DataTrainingArguments:
    data_path: Optional[str] = field(
        default="data/train_ds.csv", metadata={"help": "Path to data"}
    )

    class_weight_path: Optional[str] = field(
        default="output/class_weight.npy",
        metadata={"help": "Path to store class_weight"},
    )

def runner(
    data_path:str,
    class_weight_path: str,
    config_dir: str,
    model_name_or_path: str,
    cache_dir: str,
    freeze_feature: bool=False,
    num_classes: int = 2,
    act_func: str="sigmoid"
):
    config = read_yaml_file(config_dir)["segment_kits"]
    class_weight = np.array([0.25, 0.75])  
    class_weight = class_weight.reshape(1, num_classes, *[1] * 3)
    np.save(class_weight_path, class_weight)
    logger.info("class weight saved at %s", class_weight_path)

    model = get_model(
        model=model_maps[model_name_or_path],
        pretrained_path=cache_dir,
        freeze_feature=freeze_feature,
        num_classes=num_classes,
        act=act_func,
    )

    dataset = torch.load(data_path)
    processor = DataProcessor(dataset, batch_size=4, workers=2)
    dataloader = processor.get_dloader()

    # train_minibatch = next(iter(dataloader))[0]
    # print("dataloader shape: ", train_minibatch.shape)
    # output = model(train_minibatch)
    # print("predict shape:", output.shape)

    trainer = ConfigTrainer(data_loaders=dataloader, model=model, config=config)
    trainer.train()

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    runner(
        data_path=data_args.data_path,
        class_weight_path=data_args.class_weight_path,
        config_dir=model_args.config_dir,
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        freeze_feature=model_args.freeze_feature,
        num_classes=model_args.num_classes,
        act_func=model_args.act_func
    )

if __name__ == "__main__":
    main()
