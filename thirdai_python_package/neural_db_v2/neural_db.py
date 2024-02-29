from typing import Iterable, List, Optional, Union

from chunk_stores.sqlite_chunk_store import SQLiteChunkStore
from core.chunk_store import ChunkStore
from core.documents import Document
from core.retriever import Retriever
from core.types import Chunk, NewChunkBatch
from documents import document_by_name
from retrievers.mach_retriever import MachRetriever


class NeuralDB:
    def __init__(
        self,
        chunk_store: Optional[ChunkStore] = None,
        retriever: Optional[Retriever] = None,
        **kwargs
    ):
        self.chunk_store = chunk_store or SQLiteChunkStore(**kwargs)
        self.retriever = retriever or MachRetriever(**kwargs)

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
                yield doc.chunks()

        self.insert_chunks(chunk_generator(), **kwargs)

    def search(
        self, query: str, top_k: int, constraints: dict = None, **kwargs
    ) -> List[Chunk]:
        if not constraints:
            chunk_ids = self.retriever.search([query], top_k, **kwargs)
        else:
            choices = self.chunk_store.filter_chunk_ids(constraints, **kwargs)
            chunk_ids = self.retriever.rank([query], [choices], **kwargs)
        return self.chunk_store.get_chunks(chunk_ids, **kwargs)
