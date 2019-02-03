import collections
import pathlib
import random
from random import shuffle

import numpy as np
import torch
import subprocess

NP_INT_DATATYPE = np.int

def flatten(d, parent_key='', sep='_'):
    # Logic for flatten taken from https://stackoverflow.com/a/6027615/1353861
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def grouped(iterable, n):
    # Modified from https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list/39038787
    return zip(*[iter(iterable)] * n)


# Taken from https://stackoverflow.com/questions/38191855/zero-pad-numpy-array/38192105
def padarray(A, size, const=1):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values=const)


def parse_file(file_name):
    '''Method to read the given input file and return an iterable for the lines'''
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            yield line


def get_device_id(device):
    if (device == "cpu"):
        return -1
    elif (device == "gpu"):
        return None
    else:
        return None


def shuffle_list(*ls):
    """Taken from https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order"""
    l = list(zip(*ls))
    shuffle(l)
    return zip(*l)


def chunks(l, n):
    """
    Taken from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def reverse_dict(_dict):
    return {v: k for k, v in _dict.items()}


def padarray(A, size, const=0):
    # Taken from https://stackoverflow.com/questions/38191855/zero-pad-numpy-array/38192105
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values=const).astype(NP_INT_DATATYPE)


def make_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def get_device_name(device_type):
    if torch.cuda.is_available() and "cuda" in device_type:
        return device_type
    return "cpu"

def get_current_commit_id():
    command = "git rev-parse HEAD"
    commit_id = subprocess.check_output(command.split()).strip().decode("utf-8")
    return commit_id

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    Source: https://github.com/OpenNMT/OpenNMT-py/blob/2e6935f738b5c2be26d51e3ba35c9453c77e0566/onmt/utils/misc.py#L29
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def check_id_emb(tensor, max_id, min_id=0):
    """
    check if the ids are within max id before running through Embedding layer
    :param tensor:
    :param max_id:
    :return:
    """
    assert tensor.lt(max_id).all()
    assert tensor.ge(min_id).all()

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels.long()]

def aeq(tensor_a, tensor_b):
    """
    Assert given two tensors are equal in dimensions
    :param tensor_a:
    :param tensor_b:
    :return:
    """
    assert tensor_a.dim() == tensor_b.dim()
    for d in range(tensor_a.dim()):
        assert tensor_a.size(d) == tensor_b.size(d)