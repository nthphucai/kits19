import os
from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd
import torch

from segment.data.augs import aug_maps
from segment.data.data_readers.data_reader import DatasetReader
from segment.utils.file_utils import load_json_file, logger, read_yaml_file
from segment.utils.hf_argparser import HfArgumentParser


@dataclass
class DataTrainingArguments:
    data_path: Optional[str] = field(
        default="data/train_ds.csv", metadata={"help": "Path to data"}
    )
    
    config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to config file"}
    )

    out_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to dataloader output path"}
    )

    train_file_name: Optional[str] = field(
        default="train_dataset.pt",
        metadata={"help": "The name of train file"}
    )
    
    valid_file_name: Optional[str] = field(
        default="valid_dataset.pt",
        metadata={"help": "The name of valid file"}
    )

    test_file_name: Optional[str] = field(
        default="test_dataset.pt",
        metadata={"help": "The name of test file"}
    )
    
    fold: Optional[int] = field(
        default=None,
        metadata={"help": "fold"}
    )

def get_dataset(
      data_path:str, 
      config_path:str, 
      fold:int=1, 
      out_path:str=None, 
      train_file_name:str="train_dataset.pt", 
      valid_file_name:str="valid_dataset.pt", 
      test_file_name:str="test_dataset.pt", 
    ):
    data = load_json_file(data_path)["data"]
    config = read_yaml_file(config_path)["create_dataset"]
    print("config file:\n", config)
    
    df = pd.DataFrame(data)[:10]
    logger.info(f"The number of data at {df.shape[0]}")
    train_df = df.loc[df["fold"] != fold].reset_index(drop=True)
    valid_df = df.loc[df["fold"] == fold].reset_index(drop=True)

    train_ds = DatasetReader(df=df, augs=aug_maps["transforms"], phase="train", **config)
    valid_ds = DatasetReader(df=df, augs=aug_maps["transforms"], phase="valid", **config)

    train_file_path = os.path.join(out_path, train_file_name)
    torch.save(train_ds, os.path.join(train_file_path))
    logger.info(f"Saved train dataset at {train_file_path}")

    valid_file_path = os.path.join(out_path, valid_file_name)
    torch.save(valid_ds, os.path.join(out_path, valid_file_path))
    logger.info(f"Saved valid dataset at {valid_file_path}")
    return train_ds, valid_ds

def main():
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0] 
    
    train_ds, valid_ds = get_dataset(
        data_path=data_args.data_path, 
        config_path=data_args.config_path,
        fold=data_args.fold,
        out_path=data_args.out_path,
        train_file_name=data_args.train_file_name,
        valid_file_name=data_args.valid_file_name,
    ) 


if __name__ == "__main__":
    main()
