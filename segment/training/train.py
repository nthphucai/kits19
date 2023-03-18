import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch

from segment.data.data_loaders.processor import DataProcessor
from segment.models import model_maps
from segment.models.segment import get_model
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
        default="softmax", metadata={"help": "activate function"}
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


@dataclass
class TrainingArguments:
    num_train_epochs: Optional[int] = field(
        default=2,
        metadata={"help": "The number of epochs"},
    )

    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Whether to save log"},
    )


def runner(
    data_path: str,
    class_weight_path: str,
    config_dir: str,
    model_name_or_path: str,
    cache_dir: str,
    freeze_feature: bool = False,
    num_classes: int = 2,
    act_func: str = "sigmoid",
    num_train_epochs: str = 2,
    output_dir: str = None,
    log_dir: str = None,
    fp16: bool = False,
    fold: Optional[int] = 1,
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

    trainer = ConfigTrainer(
        data_loaders=dataloader,
        model=model,
        config=config,
        save_config_path=None,
        verbose=False,
        num_train_epochs=num_train_epochs,
        output_dir=output_dir,
        log_dir=log_dir,
        fp16=fp16,
        fold=fold,
    )
    trainer.train()

    logger.info("Save logs at directory: %s", log_dir)
    logger.info("Save model at directory: %s", output_dir)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    runner(
        data_path=data_args.data_path,
        class_weight_path=data_args.class_weight_path,
        config_dir=model_args.config_dir,
        model_name_or_path=model_args.model_name_or_path,
        output_dir=model_args.output_dir,
        cache_dir=model_args.cache_dir,
        freeze_feature=model_args.freeze_feature,
        num_classes=model_args.num_classes,
        act_func=model_args.act_func,
        num_train_epochs=training_args.num_train_epochs,
        log_dir=training_args.log_dir,
    )


if __name__ == "__main__":
    main()
