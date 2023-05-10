import collections
import copy
import csv
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def save_logs(epoch, logs, log_path):
    file_flags = ""
    _open_args = {"newline": "\n"}
    csv_file = None
    append_header = True

    def handle_value(k):
        is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
        if isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
            return '"[%s]"' % (", ".join(map(str, k)))
        else:
            return k

    # keys = sorted(logs.keys(), reverse = True)
    keys = list(logs.keys())

    class CustomDialect(csv.excel):
        delimiter = ","

    fieldnames = ["epoch"] + keys

    """
     'a': Append mode that opens an existing file or creates a new file for appending.
    """
    if os.path.exists(log_path):
        with open(log_path, "r" + file_flags) as f:
            append_header = not bool(len(f.readline()))
        mode = "a"
    else:
        mode = "w"

    csv_file = io.open(log_path, mode + file_flags, **_open_args)
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, dialect=CustomDialect)
    if append_header:
        writer.writeheader()
    row_dict = collections.OrderedDict({"epoch": epoch})
    row_dict.update((key, handle_value(logs[key])) for key in keys)
    writer.writerow(row_dict)
    csv_file.flush()

    return csv_file, log_path


def save_model(loss, epoch, model, optimizer, model_path):
    best_epoch = epoch
    checkpoints = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": loss,
        "epoch": best_epoch,
    }

    torch.save(checkpoints, model_path)


def plot_df(results: pd.DataFrame, name=("Train, Loss, AUC")):
    results = pd.read_csv(results, index_col=0)
    """
    Plot loss results
    """
    plt.figure(figsize=(30, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results[name[1]], label=name[0])
    # plt.title('Loss')
    plt.ylabel(name[1])
    plt.xlabel("Epochs")
    plt.legend()

    """
    Plot metric results
    """
    plt.figure(figsize=(60, 5))
    plt.subplot(1, 4, 2)
    plt.plot(results[name[2]], label=name[0])
    # plt.title(' Dice Score')
    plt.ylabel(name[2])
    plt.xlabel("Epochs")
    plt.legend()
