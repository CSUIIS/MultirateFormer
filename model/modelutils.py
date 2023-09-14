from typing import Optional, Union

import numpy as np
import torch


# 可以考虑再加一个缺失率编码
def generate_original_PE(length: int, d_model: int) -> torch.Tensor:
    """Generate positional encoding as described in original paper.  :class:`torch.Tensor`

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))

    return PE


def generate_regular_PE(length: int, d_model: int, period: Optional[int] = 24) -> torch.Tensor:
    """Generate positional encoding with a given period.

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.
    period:
        Size of the pattern to repeat.
        Default is 24.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))

    return PE


def generate_local_map_mask(chunk_size: int,
                            attention_size: int,
                            mask_future=False,
                            device: torch.device = 'cpu') -> torch.BoolTensor:
    """Compute attention mask as attention_size wide diagonal.

    Parameters
    ----------
    chunk_size:
        Time dimension size.
    attention_size:
        Number of backward elements to apply attention.
    device:
        torch device. Default is ``'cpu'``.

    Returns
    -------
        Mask as a boolean tensor.
    """
    local_map = np.empty((chunk_size, chunk_size))
    i, j = np.indices(local_map.shape)

    if mask_future:
        local_map[i, j] = (i - j > attention_size) ^ (j - i > 0)
    else:
        local_map[i, j] = np.abs(i - j) > attention_size

    return torch.BoolTensor(local_map).to(device)


def generate_sampling_PE(mask_map):

    #  mask_map:[batch, K, d_input]
    length = mask_map.shape[1]
    batch_size = mask_map.shape[0]
    types = torch.sum(mask_map, dim=-1)    # [batch， K]
    index = torch.zeros_like(types)
    for i in range(batch_size):
        index[i, :] = torch.arange(length)
    # print(index)
    # print(torch.fmod(index , 6))
    for i in range(batch_size):
        if types[i, 0] == 0 and types[i, 1] == 29:
            index[i, :] = torch.fmod(index[i, :], 6)

        if types[i, 0] == 29 and types[i, 1] == 14:
            index[i, :] = torch.fmod(index[i, :] + 1, 6)

        if types[i, 0] == 14 and types[i, 1] == 15:
            index[i, :] = torch.fmod(index[i, :] + 2, 6)

        if types[i, 0] == 15 and types[i, 1] == 14:
            index[i, :] = torch.fmod(index[i, :] + 3, 6)

        if types[i, 0] == 14 and types[i, 1] == 29:
            index[i, :] = torch.fmod(index[i, :] + 4, 6)

        if types[i, 0] == 29 and types[i, 1] == 0:
            index[i, :] = torch.fmod(index[i, :] + 5, 6)
    index = index.long()
    return index






