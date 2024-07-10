from typing import Iterable, List, Set, Tuple

from thirdai import search

from ..core.retriever import Retriever
from ..core.types import ChunkBatch, ChunkId, Score, SupervisedBatch


class FinetunableRetriever(Retriever):
    def __init__(self, on_disk=False, **kwargs):
        super().__init__()
        import uuid
        save_path = None
        if on_disk:
            save_path = f"{uuid.uuid4()}.db"
        self.retriever = search.FinetunableRetriever(save_path=save_path)

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

    def insert(self, chunks: Iterable[ChunkBatch], **kwargs):
        print("starting retriever insert")
        import time
        time.sleep(3)
        print("starting retriever for loop")
        print(len(chunks))
        for batch in chunks:
            time.sleep(3)
            print("indexing call")
            texts = batch.keywords + " " + batch.text
            time.sleep(3)

            print("CREATING IDS")
            ids = batch.chunk_id.to_list()
            time.sleep(3)

            print("CREATING TEXTS")
            texts = texts.to_list()
            time.sleep(3)

            self.retriever.index(ids=ids, docs=texts.to_list())

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
