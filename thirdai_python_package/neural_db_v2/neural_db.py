from typing import Iterable, List, Optional, Union

from chunk_stores import chunk_store_by_name
from core.documents import Document
from core.retriever import Retriever
from core.types import NewChunk
from documents import document_by_name
from retrievers import retriever_by_name

from thirdai_python_package.neural_db_v2.core.chunk_store import ChunkStore


class NeuralDB:
    def __init__(
        self,
        chunk_store: Optional[Union[ChunkStore, str]] = "default",
        retriever: Optional[Union[Retriever, str]] = "default",
        **kwargs
    ):
        self.chunk_store = (
            chunk_store
            if isinstance(chunk_store, ChunkStore)
            else chunk_store_by_name(chunk_store, **kwargs)
        )
        self.retriever = (
            retriever
            if isinstance(retriever, Retriever)
            else retriever_by_name(retriever, **kwargs)
        )

    def insert_chunks(self, chunks: Iterable[NewChunk], **kwargs):
        stored_chunks = self.chunk_store.insert(
            chunks=chunks,
            assign_new_unique_ids=True,
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
        self.insert_chunks([chunk for doc in docs for chunk in doc.chunks()], **kwargs)

    def search(self, query: str, top_k: int, constraints: dict = None, **kwargs):
        if not constraints:
            return self.retriever.search([query], top_k, **kwargs)
        choices = self.chunk_store.filter_chunk_ids(constraints, **kwargs)
        return self.retriever.rank([query], [choices], **kwargs)
