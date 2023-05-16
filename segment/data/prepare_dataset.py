import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from segment.data.augs import aug_maps
from segment.data.data_readers.data_reader import DatasetReader
from segment.utils.file_utils import load_json_file, logger, read_yaml_file
from segment.utils.hf_argparser import HfArgumentParser


@dataclass
class DataTrainingArguments:
    data_path: Optional[str] = field(
        default="data/train_ds.csv", metadata={"help": "Path for data directory"}
    )

    config_path: Optional[str] = field(
        default=None, metadata={"help": "Path for config file directory"}
    )

    out_path: Optional[str] = field(
        default=None, metadata={"help": "Path for dataloader output directory"}
    )

    train_file_name: Optional[str] = field(
        default="train_dataset.pt", metadata={"help": "Name for train dataset"}
    )

    valid_file_name: Optional[str] = field(
        default="valid_dataset.pt", metadata={"help": "Name for valid dataset"}
    )

    test_file_name: Optional[str] = field(
        default="test_dataset.pt", metadata={"help": "Name for test dataset"}
    )

    fold: Optional[int] = field(default=None, metadata={"help": "Fold number"})


def get_dataset(
    data_path: str,
    config_path: str,
    fold: int = 1,
    out_path: str = None,
    train_file_name: str = "train_dataset.pt",
    valid_file_name: str = "valid_dataset.pt",
):
    data = load_json_file(data_path)["data"]
    config = read_yaml_file(config_path)["create_dataset"]
    print("config file:\n", config)

    df = pd.DataFrame(data)
    logger.info(f"The number of data at {df.shape[0]}")

    if fold is not None:
        train_df = df.loc[df["fold"] != fold].reset_index(drop=True)
        valid_df = df.loc[df["fold"] == fold].reset_index(drop=True)
    else:
        train_df, valid_df = train_test_split(df, test_size=0.2)

    train_ds = DatasetReader(
        df=train_df, augs=aug_maps["transforms"], phase="train", **config
    )
    valid_ds = DatasetReader(df=valid_df, augs=None, phase="valid", **config)

    train_file_path = os.path.join(out_path, train_file_name)
    torch.save(train_ds, train_file_path)
    logger.info(f"The number of train dataset is {train_df.shape[0]}")
    logger.info(f"Saved train dataset at {train_file_path}")

    valid_file_path = os.path.join(out_path, valid_file_name)
    torch.save(valid_ds, valid_file_path)
    logger.info(f"The number of valid dataset is {valid_df.shape[0]}")
    logger.info(f"Saved valid dataset at {valid_file_path}")
    return train_ds, valid_ds


def main():
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    get_dataset(
        data_path=data_args.data_path,
        config_path=data_args.config_path,
        fold=data_args.fold,
        out_path=data_args.out_path,
        train_file_name=data_args.train_file_name,
        valid_file_name=data_args.valid_file_name,
    )


if __name__ == "__main__":
    main()
