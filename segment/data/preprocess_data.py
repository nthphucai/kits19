import glob
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from segment.data import create_df
from segment.utils.hf_argparser import HfArgumentParser
from segment.utils.file_utils import load_json_file, read_yaml_file, write_json_file
from segment.data.preprocess.standard_preprocess import Preprocess3D
from segment.data.utils import split_data


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
        metadata={"help": "whether to use kfold"},
    )


def main():
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    if ".csv" in pathlib.Path(data_args.data_path).suffix:
        columns = ["case_id", "img_path", "seg_path"]
        data = pd.read_csv(data_args.data_path)[columns]
    else:
        data = create_df.df_image_mask_path(root_path=data_args.data_path)
        save_path = os.path.join("data_args.data_path", "train_ds.csv")
        logger.info("csv file save at %s", save_path)
        data.to_csv(save_path)

    data_dict = data.to_dict(orient="records")[:4]
    configs = read_yaml_file(data_args.config_path)["preprocess"]

    preprocess = Preprocess3D(
        data=data_dict,
        vol_path=data_args.train_vol_path,
        seg_path=data_args.train_seg_path,
        configs=configs,
    )
    result = preprocess.run()

    if data_args.split_kfold is not None:
        result = split_data(data=result, n_split=data_args.split_kfold)

    write_json_file({"data": result}, data_args.save_file_path)
    logger.info("data path save at %s", data_args.save_file_path)


if __name__ == "__main__":
    main()
