from ..core.retriever import Retriever
from ..core.types import ChunkBatch, ChunkId, Score, SupervisedBatch
from typing import Iterable, List, Tuple

from thirdai import bolt, data


class ChunkBatchColumnMapIterator(data.PyColumnMapIterator):
    def __init__(self, iterable: Iterable[ChunkBatch]):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def next(self):
        for batch in self.chunk_iterator:
            yield data.ColumnMap(
                {
                    Mach.WEAK: data.columns.StringColumn(batch.text),
                    Mach.STRONG: data.columns.StringColumn(batch.keywords),
                    Mach.ID: data.columns.DecimalColumn(batch.chunk_id),
                }
            )
        return None

    def resource_name(self):
        return "ChunkBatchIterable"

    def restart(self):
        self.iterator = iter(self.iterable)


class EarlyStopWithMinEpochs(bolt.train.callbacks.Callback):
    def __init__(self, min_epochs, tracked_metric, metric_threshold):
        super().__init__()

        self.epoch_count = 0
        self.min_epochs = min_epochs
        self.tracked_metric = tracked_metric
        self.metric_threshold = metric_threshold

    def on_epoch_end(self):
        self.epoch_count += 1

        if (
            self.epoch_count > self.min_epochs
            and self.history[f"train_{self.tracked_metric}"][-1] > self.metric_threshold
        ):
            self.train_state.stop_training()


class Mach(Retriever):
    STRONG = "keywords"
    WEAK = "text"
    TEXT = "text"
    ID = "chunk_id"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = (
            bolt.MachConfig()
            .text_col("text")
            .id_col("id")
            .tokenizer("words")
            .contextual_encoding("none")
            .emb_dim(512)
            .n_buckets(10000)
            .emb_bias()
            .output_bias()
            .output_activation("sigmoid")
            .build()
        )

    def search(
        self, queries: List[str], top_k: int, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        return self.model.search(
            queries=queries,
            top_k=top_k,
            sparse_inference=kwargs.get("sparse_inference", False),
        )

    def rank(
        self, queries: List[str], choices: List[List[ChunkId]], top_k: int, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        return self.model.rank(
            queries=queries,
            candidates=choices,
            top_k=top_k,
            sparse_inference=kwargs.get("sparse_inference", False),
        )

    def upvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        self.model.upvote(queries=queries, ids=chunk_ids)

    def downvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        raise NotImplementedError("Method 'downvote' is not supported for Mach.")

    def associate(self, sources: List[str], targets: List[str], **kwargs):
        self.model.associate(
            sources=sources,
            targets=targets,
            n_buckets=kwargs.get("n_buckets", 7),
        )

    def insert(self, chunks: Iterable[ChunkBatch], **kwargs):
        train_data = ChunkBatchColumnMapIterator(chunks)

        metrics = (kwargs.get("metrics", []))
        if "hash_precision@5" not in metrics:
            metrics.append("hash_precision@5")

        early_stop_callback = EarlyStopWithMinEpochs(
            min_epochs=kwargs.get("early_stop_min_epochs", 3),
            tracked_metric=kwargs.get("early_stop_metric", "hash_precision@5"),
            metric_threshold=kwargs.get("early_stop_metric_threshold", 0.95),
        )

        self.model.coldstart(
            data=train_data,
            strong_cols=[Mach.STRONG],
            weak_cols=[Mach.WEAK],
            learning_rate=kwargs.get("learning_rate", 0.001),
            epochs=kwargs.get("epochs", 15),
            metrics=metrics,
            callbacks=[early_stop_callback],
            max_in_memory_batches=kwargs.get("max_in_memory_batches", None),
        )

    def supervised_train(self, samples: Iterable[SupervisedBatch], **kwargs):
        for batch in samples:
            train_data = data.ColumnMap(
                {
                    Mach.TEXT: data.columns.StringColumn(batch.query),
                    Mach.ID: data.columns.DecimalColumn(batch.chunk_id),
                }
            )

            self.model.train(
                data=train_data,
                learning_rate=kwargs.get("learning_rate", 0.0001),
                epochs=kwargs.get("epochs", 3),
                metrics=kwargs.get("metrics", ["hash_precision@5"]),
            )

    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        self.model.erase(ids=chunk_ids)
