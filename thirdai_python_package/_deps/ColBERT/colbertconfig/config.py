from dataclasses import dataclass

from .base_config import BaseConfig
from .settings import *


@dataclass
class ColBERTConfig(
    RunSettings,
    ResourceSettings,
    DocSettings,
    QuerySettings,
    TrainingSettings,
    IndexingSettings,
    SearchSettings,
    BaseConfig,
):
    pass
