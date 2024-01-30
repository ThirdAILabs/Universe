from typing import Union, Optional, Iterable
from pathlib import Path

from core.types import Document
from core.retriever import Retriever
from core.index import Index
from utils.checkpointing import prepare_checkpoint_location
from utils.kwarg_processing import extract_kwargs
from retrievers import retriever_by_name
from indexes import index_by_name


class NeuralDB:
    @staticmethod
    def _index_kwargs(kwargs):
        return extract_kwargs(kwargs, prefix="index_")

    @staticmethod
    def _retriever_kwargs(kwargs):
        return extract_kwargs(kwargs, prefix="retriever_")

    def __init__(
        self,
        index: Optional[Union[Index, str]] = "default",
        retriever: Optional[Union[Retriever, str]] = "default",
        **kwargs
    ):
        self.index = (
            index
            if isinstance(index, Index)
            else index_by_name(index, **NeuralDB._index_kwargs(kwargs))
        )
        self.retriever = (
            retriever
            if isinstance(retriever, Retriever)
            else retriever_by_name(retriever, **NeuralDB._retriever_kwargs(kwargs))
        )

    def insert(
        self, docs: Iterable[Document], checkpoint: Optional[Union[Path, str]], **kwargs
    ):
        checkpoint = checkpoint and Path(checkpoint)
        prepare_checkpoint_location(checkpoint)
        self.index.insert_batch(
            docs,
            assign_new_unique_ids=True,
            checkpoint=checkpoint and checkpoint / "index",
            **NeuralDB._index_kwargs(kwargs),
        )
        self.retriever.insert_batch(
            docs,
            checkpoint=checkpoint and checkpoint / "retriever",
            **NeuralDB._retriever_kwargs(kwargs),
        )

    def find(self, query: str, top_k: int, constraints: dict = None, **kwargs):
        if not constraints:
            return self.retriever.find(
                query, top_k, **NeuralDB._retriever_kwargs(kwargs)
            )
        choices = self.index.matching_doc_ids(
            constraints, **NeuralDB._index_kwargs(kwargs)
        )
        return self.retriever.rank(query, choices, **NeuralDB._retriever_kwargs(kwargs))
