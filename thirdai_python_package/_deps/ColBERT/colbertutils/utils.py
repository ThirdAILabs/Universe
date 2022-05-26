import os
import torch
import datetime


def timestamp(daydir=False):
    format_str = f"%Y-%m{'/' if daydir else '-'}%d{'/' if daydir else '_'}%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result


def torch_load_dnn(path):
    return torch.load(path, map_location="cpu")
