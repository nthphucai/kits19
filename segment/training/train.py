from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from segment.data.data_loaders import get_dloader
from segment.models.segment import model_segment
from segment.training.trainer.config_trainer import ConfigTrainer
from segment.utils.file_utils import read_yaml_file
from segment.utils.hf_argparser import HfArgumentParser


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

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data = pd.read_csv(data_args.data_path, index_col=0)[:1]
    model = model_segment(
        pretrained_path=model_args.model_path, freeze_feature=False, act="softmax"
    )

    config = read_yaml_file("configs/segment_pipeline.yaml")["segment_kits"]
    data = get_dloader(df=data, fold=None)

    trainer = ConfigTrainer(data=data, model=model, config=config)
    trainer.train()


if __name__ == "__main__":
    main()
