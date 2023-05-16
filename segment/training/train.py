import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

from segment.models import model_maps
from segment.models.segment import get_model
from segment.utils.file_utils import logger, read_yaml_file
from segment.training.trainer.config_trainer import ConfigTrainer
from segment.utils.hf_argparser import HfArgumentParser


@dataclass
class ModelArguments:

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path for pretrained model or model"},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to load the pretrained models"},
    )

    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models"},
    )

    config_dir: Optional[str] = field(
        default="configs/segment_pipeline.yaml",
        metadata={"help": "Path for config file directory"},
    )

    freeze_feature: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze_feature"},
    )

    act_func: Optional[str] = field(
        default="softmax", metadata={"help": "activate function type"}
    )


@dataclass
class DataTrainingArguments:
    train_dataset_path: Optional[str] = field(
        default="output/data/train_dataset.pt", metadata={"help": "Path to train dataset"}
    )

    valid_dataset_path: Optional[str] = field(
        default="output/data/valid_dataset.pt", metadata={"help": "Path to valid dataset"}
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

    do_train: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to train model"},
    )

    do_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to eval model"},
    )

    per_device_train_batch_size: int = field(
        default=2, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=2, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )


def runner(
    train_dataset_path: str,
    valid_dataset_path: Optional[str],
    class_weight_path: str,
    config_dir: str,
    model_name_or_path: str,
    cache_dir: Optional[str],
    freeze_feature: bool = False,
    num_train_epochs: str = 2,
    out_dir: str = None,
    log_dir: str = None,
    fp16: bool = False,
    do_train: bool = True,
    do_eval: bool = False,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
):

    if not os.path.exists(os.path.dirname(log_dir)):
        os.makedirs(os.path.dirname(log_dir))

    config = read_yaml_file(config_dir)["segment_kits"]
    num_classes = config["model"]["num_classes"]
    act_func = config["model"]["act_func"]
    
    class_weight = np.array([0.1, 0.3, 0.6]) if num_classes == 3 else np.array([0.25, 0.75])
    class_weight = class_weight.reshape(1, num_classes, *[1] * 3)
    np.save(class_weight_path, class_weight)

    model = get_model(
        model=model_maps[model_name_or_path],
        pretrained_path=cache_dir,
        freeze_feature=freeze_feature,
        num_classes=num_classes,
        act=act_func,
    )

    train_dataset = torch.load(train_dataset_path)
    valid_dataset = torch.load(valid_dataset_path) if valid_dataset_path is not None else None
    
    logger.info("The number of train samples: %s", len(train_dataset))
    logger.info("The number of eval samples: %s", len(valid_dataset))

    trainer = ConfigTrainer(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        model=model,
        config=config,
        save_config_path=None,
        verbose=False,
        num_train_epochs=num_train_epochs,
        out_dir=out_dir,
        log_dir=log_dir,
        fp16=fp16,
        do_train=do_train,
        do_eval=do_eval,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
    )
    trainer.train()

    logger.info("Save logs at directory: %s", log_dir)
    logger.info("Save class weight at directory %s", class_weight_path)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert isinstance(model_args, object)
    runner(
        train_dataset_path=data_args.train_dataset_path,
        valid_dataset_path=data_args.valid_dataset_path,
        class_weight_path=data_args.class_weight_path,
        config_dir=model_args.config_dir,
        model_name_or_path=model_args.model_name_or_path,
        out_dir=model_args.output_dir,
        cache_dir=model_args.cache_dir,
        freeze_feature=model_args.freeze_feature,
        act_func=model_args.act_func,
        num_train_epochs=training_args.num_train_epochs,
        log_dir=training_args.log_dir,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
    )


if __name__ == "__main__":
    main()
    