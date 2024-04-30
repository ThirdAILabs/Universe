from pathlib import Path

from .finetunable_retriever import FinetunableRetriever
from .mach import Mach

retriever_name_map = {
    "FinetunableRetriever": FinetunableRetriever,
    "Mach": Mach,
}


def load_retriever(path: Path, retriever_name: str):
    if retriever_name not in retriever_name_map:
        raise ValueError(f"Class name {retriever_name} not found in registry.")

    return retriever_name_map[retriever_name].load(path)
