from typing import Iterable, List, Set, Tuple

from thirdai import search

from ..core.retriever import Retriever
from ..core.types import ChunkBatch, ChunkId, Score, SupervisedBatch


class FinetunableRetriever(Retriever):
    def __init__(self, **kwargs):
        super().__init__()
        self.retriever = search.FinetunableRetriever()

    def search(
        self, queries: List[str], top_k: int, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        return self.retriever.query(queries, k=top_k)

    def rank(
        self, queries: List[str], choices: List[Set[ChunkId]], top_k: int, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        return self.retriever.rank(queries, candidates=choices, k=top_k)

    def upvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        self.retriever.finetune(
            doc_ids=list(map(lambda id: [id], chunk_ids)), queries=queries
        )

    def associate(
        self, sources: List[str], targets: List[str], associate_strength=4, **kwargs
    ):
        self.retriever.associate(
            sources=sources, targets=targets, strength=associate_strength
        )

    def insert(self, chunks: Iterable[ChunkBatch], index_batch_size=100000, **kwargs):
        for chunk in chunks:
            # Indexing in batches within a chunk reduces the RAM usage significantly
            # for large chunks
            for i in range(0, len(chunk), index_batch_size):
                ids = chunk.chunk_id[i : i + index_batch_size]
                texts = (
                    chunk.keywords[i : i + index_batch_size]
                    + " "
                    + chunk.text[i : i + index_batch_size]
                )
                self.retriever.index(ids=ids.to_list(), docs=texts.to_list())

    def supervised_train(self, samples: Iterable[SupervisedBatch], **kwargs):
        for batch in samples:
            self.retriever.finetune(
                doc_ids=batch.chunk_id.to_list(), queries=batch.query.to_list()
            )

    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        self.retriever.remove(ids=chunk_ids)

    def save(self, path: str):
        self.retriever.save(path)

    @classmethod
    def load(cls, path: str):
        instance = cls()
        instance.retriever = search.FinetunableRetriever.load(path)
        return instance
