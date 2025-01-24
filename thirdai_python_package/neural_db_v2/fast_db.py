import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from thirdai import search

from .core.documents import Document
from .core.reranker import Reranker
from .core.supervised import SupervisedDataset
from .core.types import Chunk, ChunkId, InsertedDocMetadata, Score
from .documents import document_by_name
from .rerankers.pretrained_reranker import PretrainedReranker
from .retrievers.finetunable_retriever import Splade


class FastDB:
    def __init__(
        self,
        save_path: str,
        splade: bool = False,
        preload_reranker: bool = False,
        **kwargs,
    ):
        os.makedirs(save_path, exist_ok=True)

        if os.path.exists(self.config_path(save_path)):
            config = self.load_config(save_path)
            splade = config["splade"]
        else:
            self.save_config(save_path, splade=splade)

        self.db = search.OnDiskNeuralDB(save_path=save_path)

        if preload_reranker:
            self.reranker: Optional[Reranker] = PretrainedReranker()
        else:
            self.reranker: Optional[Reranker] = None

        if splade:
            self.splade = Splade()
        else:
            self.splade = None

    def insert(
        self, docs: List[Union[str, Document]], **kwargs
    ) -> List[InsertedDocMetadata]:
        docs = [
            doc if isinstance(doc, Document) else document_by_name(doc) for doc in docs
        ]

        insert_metadata = []

        for doc in docs:
            doc_id = doc.doc_id()

            doc_version = None

            doc_chunks = []

            for batch in doc.chunks():
                if batch.metadata is not None:
                    metadata = batch.metadata.to_dict(orient="records")
                else:
                    metadata = [{}] * len(batch)

                if self.splade:
                    text = pd.Series([self.splade.augment(t) for t in batch.text])
                else:
                    text = batch.text

                chunks = (text + " " + batch.keywords).to_list()
                meta = self.db.insert(
                    chunks=chunks,
                    metadata=metadata,
                    document=batch.document[0],  # This is repeated for all chunks.
                    doc_id=doc_id,
                    doc_version=doc_version,
                )
                # Ensures that if the doc has multiple batches they all have the same version.
                doc_version = meta.doc_version

                doc_chunks.extend(range(meta.start_id, meta.end_id))

            insert_metadata.append(
                InsertedDocMetadata(
                    doc_id=doc_id, doc_version=doc_version, chunk_ids=doc_chunks
                )
            )

        return insert_metadata

    def search(
        self,
        query: str,
        top_k: int = 5,
        constraints: dict = None,
        rerank: bool = False,
        **kwargs,
    ) -> List[Tuple[Chunk, Score]]:
        if self.splade:
            query = self.splade.augment(query)

        if constraints:
            results = self.db.rank(query=query, constraints=constraints, top_k=top_k)
        else:
            results = self.db.query(query=query, top_k=top_k)

        results = [
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

        if rerank:
            return self.rerank(query=query, results=results)
        return results

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

    def rerank(
        self, query: str, results: List[Tuple[Chunk, Score]]
    ) -> List[Tuple[Chunk, Score]]:
        if self.reranker is None:
            self.reranker = PretrainedReranker()
        return self.reranker.rerank(query, results)

    def delete_doc(
        self,
        doc_id: str,
        keep_latest_version: bool = False,
        return_deleted_chunks: bool = False,
    ):
        self.db.delete_doc(doc_id=doc_id, keep_latest_version=keep_latest_version)

    def upvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        self.db.finetune(queries, [[chunk_id] for chunk_id in chunk_ids], **kwargs)

    def associate(self, sources: List[str], targets: List[str], **kwargs):
        self.db.associate(sources=sources, targets=targets, **kwargs)

    def supervised_train(self, supervised: SupervisedDataset, **kwargs):
        for batch in supervised.samples():
            self.db.finetune(
                queries=batch.query.to_list(), chunk_ids=batch.chunk_id.to_list()
            )

    def documents(self) -> List[dict]:
        return self.db.sources()

    @staticmethod
    def config_path(base: str):
        return os.path.join(base, "ndb_config.json")

    @staticmethod
    def save_config(path: str, **kwargs):
        with open(FastDB.config_path(path), "w") as f:
            json.dump(kwargs, f)

    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        with open(FastDB.config_path(path), "r") as f:
            return json.load(f)

    def save(self, path: str):
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise ValueError(
                    f"Cannot save NeuralDB to {path} since it is not a directory"
                )
        else:
            os.makedirs(path)

        self.db.save(path)
        self.save_config(path, splade=self.splade is not None)

    @staticmethod
    def load(path: str):
        return FastDB(save_path=path)
