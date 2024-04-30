from .mach import Mach
from ..core.retriever import Retriever


class MachEnsemble(Retriever):
    def search(
        self, queries: List[str], top_k: int, sparse_inference: bool = False, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        return self.model.search(
            queries=queries,
            top_k=top_k,
            sparse_inference=sparse_inference,
        )

    def rank(
        self,
        queries: List[str],
        choices: List[List[ChunkId]],
        top_k: int,
        sparse_inference: bool = False,
        **kwargs,
    ) -> List[List[Tuple[ChunkId, Score]]]:
        return self.model.rank(
            queries=queries,
            candidates=choices,
            top_k=top_k,
            sparse_inference=sparse_inference,
        )

    def upvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        self.model.upvote(queries=queries, ids=chunk_ids)

    def associate(
        self, sources: List[str], targets: List[str], n_buckets: int = 7, **kwargs
    ):
        self.model.associate(
            sources=sources,
            targets=targets,
            n_buckets=n_buckets,
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
        train_data = ChunkColumnMapIterator(
            chunks, text_columns={Mach.STRONG: "keywords", Mach.WEAK: "text"}
        )

        metrics = metrics or []
        if "hash_precision@5" not in metrics:
            metrics.append("hash_precision@5")

        min_epochs, max_epochs = autotune_from_scratch_min_max_epochs(
            size=train_data.size()
        )

        early_stop_callback = EarlyStopWithMinEpochs(
            min_epochs=epochs or min_epochs,
            tracked_metric=early_stop_metric,
            metric_threshold=early_stop_metric_threshold,
        )

        callbacks = callbacks or []
        callbacks.append(early_stop_callback)

        self.model.coldstart(
            data=train_data,
            strong_cols=[Mach.STRONG],
            weak_cols=[Mach.WEAK],
            learning_rate=learning_rate,
            epochs=epochs or max_epochs,
            metrics=metrics,
            callbacks=callbacks,
            max_in_memory_batches=max_in_memory_batches,
            variable_length=variable_length,
            batch_size=batch_size,
        )

    def supervised_train(
        self,
        samples: Iterable[SupervisedBatch],
        learning_rate: float = 0.001,
        epochs: int = 3,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ):
        train_data = ChunkColumnMapIterator(samples, text_columns={Mach.TEXT: "query"})

        self.model.train(
            data=train_data,
            learning_rate=learning_rate,
            epochs=epochs,
            metrics=metrics or ["hash_precision@5"],
        )

    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        self.model.erase(ids=chunk_ids)
