import glob
import json
import logging
import os
import pathlib
import pickle
import shutil
from typing import List, Optional

import pandas as pd
import yaml

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Access to Yaml File
def read_yaml_file(file_path: str):
    with open(file_path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    return config


def read_df_file(file_path: str):
    return pd.read_csv(file_path)


# access to text files
def read_text_file(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        context = [item.strip() for item in f.readlines()]
    return context


def write_text_file(context: List[str], file_path: str):
    with open(file_path, "w") as f:
        for text in context:
            f.write(text + "\n")


# remove all files in path
def remove_files(path):
    files = glob.glob(f"{path}/*")
    for f in files:
        os.remove(f)


# access to Picke File
def write_pickle_file(data: dict, path: str, name: Optional[str] = None) -> None:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".pkl")
    else:
        save_path = path

    f = open(save_path, "wb")
    pickle.dump(data, f)
    f.close()


def read_pickle_file(path, name: Optional[str] = None) -> dict:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".pkl")
    else:
        save_path = path

    f = open(save_path, "rb")
    pickle_file = pickle.load(f)
    return pickle_file


# access to Json File
def write_json_file(
    data: dict, path: str, name: Optional[str] = None, **kwargs
) -> None:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".json")
    else:
        save_path = path
    with open(save_path, "w") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4, **kwargs)


def load_json_file(path: str, name: Optional[str] = None, **kwargs) -> dict:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".json")
    else:
        save_path = path
    with open(save_path, encoding="utf-8") as outfile:
        data = json.load(outfile, **kwargs)
        return data


def load_dataframe_file(path: str, convert_to_json: bool = True):
    file_extension = pathlib.Path(path).suffix
    if ".xlsx" in file_extension:
        data = pd.read_excel(path, sheet_name=["Template"])
    elif ".csv" in file_extension:
        data = pd.read_csv(path)
    else:
        raise NotImplementedError("only support .csv and .xlsx extension")

    if convert_to_json:
        data = pd.DataFrame(data["context"])
        data.dropna(inplace=True, how="all")
        data = pd.DataFrame(pd.unique(data["context"]), data=["context"])
        data = data.reset_index(drop=True)
        data = data.to_dict("records")
    return data


# def download_trained_model(
#     save_path: Optional[str] = "/tmp/quesgen_model", task: str = "multitask"
# ):
#     """
#     Download multitask model or multichoice model of QuesGen
#     """
#     if save_path is None:
#         logger.warning(f"save_path is not specified, default at {save_path}")
#     if task == "multitask":
#         os.system(
#             f"wget --progress=bar:force:noscroll {MULTITASK_MODEL_URL} -P {save_path}"
#         )
#         shutil.unpack_archive(
#             filename=f"{save_path}/{MULTITASK_MODEL}" + ".zip", extract_dir=save_path
#         )
#         logger.info(f"{MULTITASK_MODEL} downloaded!")
#     elif task == "multichoice":
#         os.system(
#             f"wget --progress=bar:force:noscroll {MULTICHOICE_MODEL_URL} -P {save_path}"
#         )
#         shutil.unpack_archive(
#             filename=f"{save_path}/{MULTICHOICE_MODEL}" + ".zip", extract_dir=save_path
#         )
#         logger.info(f"{MULTICHOICE_MODEL} downloaded!")
#     else:
#         raise NotImplementedError("only support multitask and multiplechoice task")
