import json
import os
from typing import Iterable, List, Optional, Tuple, Union

from .chunk_stores import PandasChunkStore, SQLiteChunkStore
from .core.chunk_store import ChunkStore
from .core.documents import Document
from .core.retriever import Retriever
from .core.supervised import SupervisedDataset
from .core.types import Chunk, ChunkId, CustomIdSupervisedBatch, NewChunkBatch, Score
from .documents import document_by_name
from .retrievers import FinetunableRetriever, Mach


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
        self, query: str, top_k: int, constraints: dict = None, **kwargs
    ) -> List[Tuple[Chunk, Score]]:
        return self.search_batch([query], top_k, constraints, **kwargs)[0]

    def search_batch(
        self, queries: List[str], top_k: int, constraints: dict = None, **kwargs
    ) -> List[List[Tuple[Chunk, Score]]]:
        if not constraints:
            results = self.retriever.search(queries, top_k, **kwargs)
        else:
            choices = self.chunk_store.filter_chunk_ids(constraints, **kwargs)
            # TODO is there a better way that duplicating the constraints here
            results = self.retriever.rank(queries, [choices for _ in queries], **kwargs)

        chunk_results = []
        for query_results in results:
            chunk_ids, scores = [list(tup) for tup in zip(*query_results)]
            chunks = self.chunk_store.get_chunks(chunk_ids)
            chunk_results.append(list(zip(chunks, scores)))

        return chunk_results

    def delete(self, chunk_ids: List[ChunkId]):
        self.chunk_store.delete(chunk_ids)
        self.retriever.delete(chunk_ids)

    def upvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        self.retriever.upvote(queries, chunk_ids, **kwargs)

    def associate(self, sources: List[str], targets: List[str], **kwargs):
        self.retriever.associate(sources, targets, **kwargs)

    def supervised_train(self, supervised: SupervisedDataset, **kwargs):
        iterable = supervised.samples()

        if isinstance(next(iter(iterable)), CustomIdSupervisedBatch):
            iterable = self.chunk_store.remap_custom_ids(iterable)

        self.retriever.supervised_train(iterable, **kwargs)

    @staticmethod
    def chunk_store_path(directory: str) -> str:
        return os.path.join(directory, "chunk_store")

    @staticmethod
    def retriever_path(directory: str) -> str:
        return os.path.join(directory, "retriever")

    @staticmethod
    def metadata_path(directory: str) -> str:
        return os.path.join(directory, "metadata.json")

    @staticmethod
    def load_chunk_store(path: str, chunk_store_name: str):
        chunk_store_name_map = {
            "PandasChunkStore": PandasChunkStore,
            "SQLiteChunkStore": SQLiteChunkStore,
        }
        if chunk_store_name not in chunk_store_name_map:
            raise ValueError(f"Class name {chunk_store_name} not found in registry.")

        return chunk_store_name_map[chunk_store_name].load(path)

    @staticmethod
    def load_retriever(path: str, retriever_name: str):
        retriever_name_map = {
            FinetunableRetriever.__name__: FinetunableRetriever,
            Mach.__name__: Mach,
        }
        if retriever_name not in retriever_name_map:
            raise ValueError(f"Class name {retriever_name} not found in registry.")

        return retriever_name_map[retriever_name].load(path)

    def save(self, path: str):
        os.makedirs(path)

        self.chunk_store.save(self.chunk_store_path(path))
        self.retriever.save(self.retriever_path(path))

        metadata = {
            "chunk_store_name": self.chunk_store.__class__.__name__,
            "retriever_name": self.retriever.__class__.__name__,
        }

        with open(self.metadata_path(path), "w") as f:
            json.dump(metadata, f)

    @staticmethod
    def load(path: str):
        with open(NeuralDB.metadata_path(path), "r") as f:
            metadata = json.load(f)

        chunk_store = NeuralDB.load_chunk_store(
            NeuralDB.chunk_store_path(path),
            chunk_store_name=metadata["chunk_store_name"],
        )
        retriever = NeuralDB.load_retriever(
            NeuralDB.retriever_path(path),
            retriever_name=metadata["retriever_name"],
        )

        return NeuralDB(chunk_store=chunk_store, retriever=retriever)
