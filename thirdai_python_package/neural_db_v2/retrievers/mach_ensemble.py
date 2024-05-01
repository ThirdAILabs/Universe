from .mach import Mach
from ..core.retriever import Retriever
from typing import List, Tuple, Optional, Iterable
from ..core.types import (
    ChunkId,
    Score,
    ChunkBatch,
    SupervisedBatch,
)
from thirdai import bolt, data


class MachEnsemble(Retriever):
    retrievers: List[Mach]

    def __init__(self, n_models, **kwargs):
        # TODO(nicholas) seeds?
        self.retrivers = [Mach(**kwargs) for _ in range(n_models)]

    def search(
        self, queries: List[str], top_k: int, sparse_inference: bool = False, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        pass

    def rank(
        self,
        queries: List[str],
        choices: List[List[ChunkId]],
        top_k: int,
        sparse_inference: bool = False,
        **kwargs,
    ) -> List[List[Tuple[ChunkId, Score]]]:
        pass

    def upvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        for retriever in self.retrievers:
            retriever.upvote(queries=queries, chunk_ids=chunk_ids, **kwargs)

    def associate(
        self, sources: List[str], targets: List[str], n_buckets: int = 7, **kwargs
    ):
        for retriever in self.retrievers:
            retriever.associate(
                sources=sources, targets=targets, n_buckets=n_buckets, **kwargs
            )

    def insert(
        self,
        chunks: Iterable[ChunkBatch],
        learning_rate: float = 0.001,
        epochs: Optional[int] = None,
        metrics: Optional[List[str]] = None,
        callbacks: Optional[List[bolt.train.callbacks.Callback]] = None,
        max_in_memory_batches: Optional[int] = None,
        variable_length: Optional[
            data.transformations.VariableLengthConfig
        ] = data.transformations.VariableLengthConfig(),
        batch_size: int = 2000,
        early_stop_metric: str = "hash_precision@5",
        early_stop_metric_threshold: float = 0.95,
        **kwargs,
    ):
        for retriever in self.retrievers:
            retriever.insert(
                chunks=chunks,
                learning_rate=learning_rate,
                epochs=epochs,
                metrics=metrics,
                callbacks=callbacks,
                max_in_memory_batches=max_in_memory_batches,
                variable_length=variable_length,
                batch_size=batch_size,
                early_stop_metric=early_stop_metric,
                early_stop_metric_threshold=early_stop_metric_threshold,
                **kwargs,
            )

    def supervised_train(
        self,
        samples: Iterable[SupervisedBatch],
        learning_rate: float = 0.001,
        epochs: int = 3,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ):
        for retriever in self.retrievers:
            retriever.supervised_train(
                samples=samples,
                learning_rate=learning_rate,
                epochs=epochs,
                metrics=metrics,
                **kwargs,
            )

    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        for retriever in self.retrievers:
            retriever.delete(chunk_ids=chunk_ids)
