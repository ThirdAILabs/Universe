from typing import List, Optional, Union, Iterable

from core.documents import Document
from core.retriever import Retriever
from core.chunk_store import ChunkStore
from core.types import NewChunkBatch
from documents import document_by_name

from chunk_stores.dataframe_chunk_store import DataFrameChunkStore
from retrievers.mach_retriever import MachRetriever


class NeuralDB:
    def __init__(
        self,
        chunk_store: Optional[ChunkStore] = None,
        retriever: Optional[Retriever] = None,
        **kwargs
    ):
        self.chunk_store = chunk_store or DataFrameChunkStore(**kwargs)
        self.retriever = retriever or MachRetriever(**kwargs)

    def insert_chunks(self, chunks: Iterable[NewChunkBatch], **kwargs):
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
