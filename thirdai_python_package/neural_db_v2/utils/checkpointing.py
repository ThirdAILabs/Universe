import os
from pathlib import Path
from typing import Optional


def prepare_checkpoint_location(path: Optional[Path]):
    if path and not os.path.exists(path):
        os.mkdir(path)
