from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import pandas as pd

from segment.utils.hf_argparser import HfArgumentParser
from segment.utils.file_utils import logger, load_json_file
from segment.data.data_loaders.data_loader import Repos
from segment.data.augs import aug_maps

@dataclass
class DataTrainingArguments:
    data_path: Optional[str] = field(
        default="data/train_ds.csv", metadata={"help": "Path to data"}
    )

    out_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to dataloader output path"}
    )

    fold: Optional[int] = field(
        default=None,
        metadata={"help": "fold"}
    )
def main():
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0] 

    data = load_json_file(data_args.data_path)["data"]
    df = pd.DataFrame(data)
    dataloader = Repos.get_dloader(df=df, augs=aug_maps["transforms"], fold=data_args.fold)

    torch.save(dataloader, data_args.out_path)
    logger.info(f"Saved dataloader at {data_args.out_path}")
 

if __name__ == "__main__":
    main()
