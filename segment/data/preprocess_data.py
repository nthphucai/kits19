import glob
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from segment.data import create_df
from segment.data.preprocess.standard_preprocess import Preprocess3D
from segment.data.utils import split_data
from segment.utils.file_utils import (load_json_file, logger, read_yaml_file,
                                      write_json_file)
from segment.utils.hf_argparser import HfArgumentParser


@dataclass
class DataTrainingArguments:
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data from ..."},
    )

    train_vol_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to train volume from ..."},
    )

    train_seg_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to train seg from ..."},
    )

    save_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to file path from ..."},
    )

    config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to config file from ..."},
    )

    split_kfold: Optional[int] = field(
        default=None,
        metadata={"help": "Whether to use kfold"},
    )

def preprocess_volume(
      data_path: str,
      config_path: str,
      train_vol_path: str,
      train_seg_path: str,
      save_file_path: str,
      split_kfold: Optional[int]=None,
    ):
      
    if ".csv" in pathlib.Path(data_path).suffix:
        columns = ["case_id", "img_path", "seg_path"]
        data = pd.read_csv(data_path)[columns]
    else:
        data = create_df.df_image_mask_path(root_path=data_path)
        save_path = os.path.join(data_path, "train_ds.csv")
        logger.info("csv file save at %s", save_path)
        data.to_csv(save_path)

    data_dict = data.to_dict(orient="records")[:10]
    configs = read_yaml_file(config_path)["preprocess"]

    preprocess = Preprocess3D(
        data=data_dict,
        vol_path=train_vol_path,
        seg_path=train_seg_path,
        configs=configs,
    )
    result = preprocess.run()
    # result = preprocess.to_dict()

    if split_kfold is not None:
        result = split_data(data=result, n_split=split_kfold)

    write_json_file({"data": result}, save_file_path)
    logger.info("data path save at %s", save_file_path)

def main():
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    preprocess_volume(
      data_path=data_args.data_path,
      config_path=data_args.config_path,
      train_vol_path=data_args.train_vol_path,
      train_seg_path=data_args.train_seg_path,
      save_file_path=data_args.save_file_path,
      split_kfold=data_args.split_kfold,

    )


if __name__ == "__main__":
    main()
