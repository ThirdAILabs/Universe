import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

from .chunk_stores import load_chunk_store
from .chunk_stores.sqlite_chunk_store import SQLiteChunkStore
from .core.chunk_store import ChunkStore
from .core.documents import Document
from .core.retriever import Retriever
from .core.supervised import Supervised
from .core.types import Chunk, ChunkId, CustomIdSupervisedBatch, NewChunkBatch
from .documents import document_by_name
from .retrievers import load_retriever
from .retrievers.mach import Mach


class NeuralDB:
    def __init__(
        self,
        chunk_store: Optional[ChunkStore] = None,
        retriever: Optional[Retriever] = None,
        **kwargs,
    ):
        self.chunk_store = chunk_store or SQLiteChunkStore(**kwargs)
        self.retriever = retriever or Mach(**kwargs)

    def insert_chunks(self, chunks: Iterable[NewChunkBatch], **kwargs):
        stored_chunks = self.chunk_store.insert(
            chunks=chunks,
            **kwargs,
        )
        self.retriever.insert(
            chunks=stored_chunks,
            **kwargs,
        )

    def insert(self, docs: List[Union[str, Document]], **kwargs):
        docs = [
            doc if isinstance(doc, Document) else document_by_name(doc) for doc in docs
        ]

        def chunk_generator():
            for doc in docs:
                for chunk in doc.chunks():
                    yield chunk

        self.insert_chunks(chunk_generator(), **kwargs)

    def search(
        self, queries: List[str], top_k: int, constraints: dict = None, **kwargs
    ) -> List[List[Chunk]]:
        if not constraints:
            chunk_ids = self.retriever.search(queries, top_k, **kwargs)
        else:
            choices = self.chunk_store.filter_chunk_ids(constraints, **kwargs)
            chunk_ids = self.retriever.rank(queries, [choices], **kwargs)
        return self.chunk_store.get_chunks(chunk_ids, **kwargs)

    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        self.retriever.delete(chunk_ids, **kwargs)

    def upvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        self.retriever.upvote(queries, chunk_ids, **kwargs)

    def associate(self, sources: List[str], targets: List[str], **kwargs):
        self.retriever.associate(sources, targets, **kwargs)

    def supervised_train(self, supervised: Supervised, **kwargs):
        iterable = supervised.samples()

        if isinstance(next(iter(iterable)), CustomIdSupervisedBatch):
            iterable = self.chunk_store.remap_custom_ids(iterable)

        self.retriever.supervised_train(iterable, **kwargs)

    def save(self, path: str):
        directory = Path(path)
        os.makedirs(directory)

        self.chunk_store.save(directory / "chunk_store")
        self.retriever.save(directory / "retriever")

        metadata = {
            "chunk_store_name": self.chunk_store.__class__.__name__,
            "retriever_name": self.retriever.__class__.__name__,
        }

        metadata_path = directory / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    @staticmethod
    def load(path: str):
        directory = Path(path)

        metadata_path = directory / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        chunk_store = load_chunk_store(
            path / "chunk_store", chunk_store_name=metadata["chunk_store_name"]
        )
        retriever = load_retriever(
            path / "retriever", retriever_name=metadata["retriever_name"]
        )

        return NeuralDB(chunk_store=chunk_store, retriever=retriever)
