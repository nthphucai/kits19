import numpy as np
import torch
import torch.nn as nn

import segment.utils.parameter as para


class Normalization(nn.Module):
    def __init__(self, mean=para.mean, std=para.std):
        """
        calculate (x - mean) / std
        :param mean: mean
        :param std: std
        """
        super().__init__()

        self.mean = nn.Parameter(
            torch.tensor(mean, dtype=torch.float), requires_grad=False
        )
        self.std = nn.Parameter(
            torch.tensor(std, dtype=torch.float), requires_grad=False
        )

    def forward(self, x):
        if x.dtype != torch.float:
            x = x.float()

        mean = self.mean.view(1, -1, *[1] * len(x.shape[2:]))
        std = self.std.view(1, -1, *[1] * len(x.shape[2:]))
        return (x - mean) / std

    def extra_repr(self):
        return f"mean={self.mean}, std={self.std}"


def normalize(data, normaltype=str):
    if normaltype == "mean_std":
        data = mean_std_norm(data, mean=para.mean, std=para.std)
    elif normaltype == "range_norm":
        data = range_norm(data)
    elif normaltype == "zero_mean":
        data = zero_mean_unit_variance_norm(data)
    return data


def min_max_norm(data, eps):
    mn = data.min()
    mx = data.max()
    data_normalized = data - mn
    old_range = mx - mn + eps
    data_normalized /= old_range
    return data_normalized


"""
chua xac minh
"""


def range_norm(data, range=(0, 1), eps=1e-8):
    data_normalized = np.zeros(data.shape, dtype=data.dtype)
    data_normalized = min_max_norm(data, eps)

    data_normalized *= range[1] - range[0]
    data_normalized += range[0]
    return data_normalized


"""
da xac minh
"""


def mean_std_norm(data, mean, std):
    data_random = np.random.normal(0, 1, size=data.shape)
    data_normalized = (data - mean) / (std)
    data_normalized[data == 0] = data_random[data == 0]
    return data_normalized


"""
normalize the itensity of an nd volume based on the mean and std of nonzeor region
"""


def zero_mean_unit_variance_norm(data, epsilon=1e-8):
    pixels = data[data > 0]
    mean = pixels.mean()
    std = pixels.std() + epsilon
    data_normalized = (data - mean) / std
    data_random = np.random.normal(0, 1, size=data.shape)
    data_normalized[data == 0] = data_random[data == 0]
    return data_normalized
