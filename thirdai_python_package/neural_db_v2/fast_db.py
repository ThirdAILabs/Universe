import json
import os
from typing import Iterable, List, Optional, Tuple, Union, Any
from thirdai import search
from .core.documents import Document
from .documents import document_by_name
from .core.types import Chunk, Score


class FastDB:
    def __init__(self, save_path: str, **kwargs):
        os.makedirs(save_path)
        self.db = search.OnDiskNeuralDB(save_path=save_path)

    def insert(self, docs: List[Union[str, Document]], **kwargs) -> List[Any]:
        docs = [
            doc if isinstance(doc, Document) else document_by_name(doc) for doc in docs
        ]

        for doc in docs:
            doc_id = doc.doc_id

            for batch in doc.chunks():
                metadata = batch.metadata.to_dict(orient="records")
                chunks = (batch.text + " " + batch.keywords).to_list()
                self.db.insert(
                    batch.document[0], chunks=chunks, metadata=metadata, doc_id=doc_id()
                )

        return []

    def search(
        self,
        query: str,
        top_k: int = 5,
        constraints: dict = None,
        rerank: bool = False,
        **kwargs,
    ) -> List[Tuple[Chunk, Score]]:

        if constraints:
            results = self.db.rank(query=query, constraints=constraints, top_k=top_k)
        else:
            results = self.db.query(query=query, top_k=top_k)

        return [
            (
                Chunk(
                    text=res.text,
                    keywords="",
                    metadata=res.metadata,
                    document=res.document,
                    doc_id=res.doc_id,
                    doc_version=res.doc_version,
                    chunk_id=res.id,
                ),
                score,
            )
            for res, score in results
        ]

    def search_batch(
        self,
        queries: List[str],
        top_k: int,
        constraints: dict = None,
        rerank: bool = False,
        **kwargs,
    ) -> List[List[Tuple[Chunk, Score]]]:
        return [
            self.search(q, top_k=top_k, constraints=constraints, rerank=rerank)
            for q in queries
        ]

    # def rerank(
    #     self, query: str, results: List[Tuple[Chunk, Score]]
    # ) -> List[Tuple[Chunk, Score]]:
    #     if self.reranker is None:
    #         self.reranker = PretrainedReranker()
    #     return self.reranker.rerank(query, results)

    # def delete_doc(
    #     self,
    #     doc_id: str,
    #     keep_latest_version: bool = False,
    #     return_deleted_chunks: bool = False,
    # ):
    #     before_version = (
    #         self.chunk_store.max_version_for_doc(doc_id)
    #         if keep_latest_version
    #         else float("inf")
    #     )
    #     chunk_ids = self.chunk_store.get_doc_chunks(
    #         doc_id=doc_id, before_version=before_version
    #     )

    #     if return_deleted_chunks:
    #         chunks_to_delete = self.chunk_store.get_chunks(chunk_ids)

    #     self.retriever.delete(chunk_ids)
    #     self.chunk_store.delete(chunk_ids)

    #     if return_deleted_chunks:
    #         return chunks_to_delete
