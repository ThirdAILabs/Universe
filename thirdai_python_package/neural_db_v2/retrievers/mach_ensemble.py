import os
import re
from typing import Iterable, List, Optional, Tuple

from thirdai import bolt, data

from ..core.retriever import Retriever
from ..core.types import ChunkBatch, ChunkId, Score, SupervisedBatch
from .mach import Mach


class MachEnsemble(Retriever):
    retrievers: List[Mach]

    def __init__(self, n_models=None, retrievers=None, **kwargs):
        if not n_models and not retrievers:
            raise ValueError(
                "When constructing MachEnsemble either n_models or retrievers must be specified."
            )
        if retrievers:
            self.retrievers = retrievers
        else:
            if "index_seed" in kwargs:
                del kwargs["index_seed"]
            self.retrievers = [Mach(**kwargs, index_seed=i) for i in range(n_models)]

    def search(
        self, queries: List[str], top_k: int, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        return bolt.MachRetriever.ensemble_search(
            retrievers=[retriever.model for retriever in self.retrievers],
            queries=queries,
            top_k=top_k,
        )

    def rank(
        self,
        queries: List[str],
        choices: List[List[ChunkId]],
        top_k: int,
        sparse_inference: bool = False,
        **kwargs,
    ) -> List[List[Tuple[ChunkId, Score]]]:
        return bolt.MachRetriever.ensemble_rank(
            retrievers=[retriever.model for retriever in self.retrievers],
            queries=queries,
            candidates=choices,
            top_k=top_k,
        )

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

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for i, retriever in enumerate(self.retrievers):
            retriever.save(os.path.join(path, f"mach_{i}"))

    @classmethod
    def load(cls, path: str):
        retrievers = []
        for file in sorted(os.listdir(path)):
            if match := re.match(r"mach_(\d+)", file):
                retriever_id = int(match.group(1))
                retrievers.append((retriever_id, Mach.load(os.path.join(path, file))))

        retrievers.sort(key=lambda x: x[0])

        ids, retrievers = tuple(zip(*retrievers))

        assert ids == tuple(range(len(ids))), "Malformed retriever ids in ensemble."

        return MachEnsemble(retrievers=retrievers)
