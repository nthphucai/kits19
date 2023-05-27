import os
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd

from segment.data.preprocess.standard_preprocess import Preprocess3D
from segment.utils.file_utils import logger, read_yaml_file, write_json_file
from segment.utils.hf_argparser import HfArgumentParser
from segment.utils.utils import get_progress


@dataclass
class DataInferenceArguments:
    data_path: Optional[str] = field(
        default="/content/drive/MyDrive/Seg3D/KiTS2019/kits19/data/",
        metadata={"help": "Path to data"},
    )

    config_path: Optional[str] = field(
        default="/content/drive/MyDrive/Seg3D/KiTS2019/kits19/configs/preprocess_pipeline.yaml",
        metadata={"help": "Path to preprocess config"},
    )

    vol_path: Optional[str] = field(
        default="/content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/test_data/vol",
        metadata={"help": "Path to preprocessed volume"},
    )

    out_path: Optional[str] = field(
        default="/content/drive/MyDrive/Seg3D/KiTS2019/kits19/output/test_data.json",
        metadata={"help": "Path to preprocessed output"},
    )


def prepare_test_data(data_path: str, config_path: str, vol_path: str, out_path: str):
    preprocess_configs = read_yaml_file(config_path)["preprocess"]
    dataset_configs = read_yaml_file(config_path)["create_dataset"]

    case_id = [f"case_00{num}" for num in range(210, 300)]
    imaging_path = [
        os.path.join(data_path, f"case_00{num}/imaging.nii.gz")
        for num in range(210, 300)
    ]
    preprocessed_vol_path = [
        os.path.join(vol_path, f"case_00{num}_imaging.nii.gz")
        for num in range(210, 300)
    ]

    test_df = pd.DataFrame({"case_id": case_id, "img_path": imaging_path})
    data_dict = test_df.to_dict(orient="records")

    preprocess = Preprocess3D(
        data=data_dict,
        configs=preprocess_configs,
        vol_path=vol_path,
        seg_path=out_path,
    )

    preprocessed_test_data = list(
        map(preprocess.create_one_test_item, get_progress(data_dict))
    )
    case_id = [item["case_id"] for item in preprocessed_test_data]
    preprocessed_vol = [item["vol"] for item in preprocessed_test_data]
    [
        np.save(vol_path, vol)
        for vol_path, vol in zip(preprocessed_vol_path, preprocessed_vol)
    ]

    result = [
        {"case_id": id_, "vol_path": vol_path, "new_vol_path": new_vol_path + ".npy"}
        for id_, vol_path, new_vol_path in zip(
            case_id, imaging_path, preprocessed_vol_path
        )
    ]

    write_json_file({"data": result}, out_path)
    logger.info("Save preprocessed test data at %s", out_path)


def main():
    parser = HfArgumentParser((DataInferenceArguments))
    data_infer_args = parser.parse_args_into_dataclasses()[0]

    prepare_test_data(
        data_path=data_infer_args.data_path,
        config_path=data_infer_args.config_path,
        vol_path=data_infer_args.vol_path,
        out_path=data_infer_args.out_path,
    )


if __name__ == "__main__":
    main()
