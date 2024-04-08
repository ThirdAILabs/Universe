from typing import Iterable, List, Tuple

from thirdai import search

from ..core.retriever import Retriever
from ..core.types import ChunkBatch, ChunkId, Score, SupervisedBatch


class InvertedIndex(Retriever):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inverted_index = search.InvertedIndex()

    def search(
        self, queries: List[str], top_k: int, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        return self.inverted_index.query(queries, k=top_k)

    def rank(
        self, queries: List[str], choices: List[List[ChunkId]], top_k: int, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        """For constrained search.
        Note on method signature:
        Choices are provided as a separate argument from queries. While it may
        be safer for the function to accept pairs of (query, choices), choices
        are likely the return value of some function fn(queries) -> choices.
        Thus, there likely exist separate collections for queries and
        choices in memory. This function signature preempts the need to reshape
        these existing data structures.
        """
        return self.inverted_index.rank(queries, candidates=choices, k=top_k)

    def upvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        raise NotImplementedError(
            "Method 'upvote' is not supported for inverted index."
        )

    def downvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        raise NotImplementedError(
            "Method 'downvote' is not supported for inverted index."
        )

    def associate(self, sources: List[str], targets: List[str], **kwargs):
        raise NotImplementedError(
            "Method 'associate' is not supported for inverted index."
        )

    def insert(self, chunks: Iterable[ChunkBatch], **kwargs):
        for batch in chunks:
            texts = (
                batch.keywords.reset_index(drop=True)
                + " "
                + batch.text.reset_index(drop=True)
            )

            self.inverted_index.index(
                ids=batch.chunk_id.to_list(), docs=texts.to_list()
            )

    def supervised_train(self, samples: Iterable[SupervisedBatch], **kwargs):
        raise NotImplementedError(
            "Method 'supervised_train' is not supported for inverted index."
        )

    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        self.inverted_index.remove(ids=chunk_ids)
