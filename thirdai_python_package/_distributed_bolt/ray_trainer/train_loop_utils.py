import shutil
from typing import Dict, Optional

from ray import train

from ..utils import check_torch_installed, timed


@timed
def prepare_model(model):
    check_torch_installed()

    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "Torch process group must be initialized before calling this Function. Make sure you are using TorchConfig as Backend Config."
        )

    if dist.get_rank() == 0:
        model_to_broadcast = [model]
    else:
        model_to_broadcast = [None]

    device = torch.device("cpu")
    dist.broadcast_object_list(model_to_broadcast, src=0, device=device)
    return model_to_broadcast[0]
