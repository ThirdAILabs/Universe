from typing import Iterable, List, Optional, Tuple

from thirdai import bolt, data

from thirdai_python_package.neural_db.models.mach_defaults import (
    autotune_from_scratch_min_max_epochs,
)

from ..core.retriever import Retriever
from ..core.types import ChunkBatch, ChunkId, Score, SupervisedBatch


class ChunkColumnMapIterator(data.ColumnMapIterator):
    def __init__(self, iterable: Iterable[ChunkBatch]):
        data.ColumnMapIterator.__init__(self)

        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def next(self) -> Optional[data.ColumnMap]:
        try:
            batch = next(self.iterator)
            return data.ColumnMap(
                {
                    Mach.WEAK: data.columns.StringColumn(batch.text),
                    Mach.STRONG: data.columns.StringColumn(batch.keywords),
                    Mach.ID: data.columns.TokenArrayColumn(
                        [[x] for x in batch.chunk_id], 4294967295
                    ),
                }
            )
        except StopIteration:
            return None

    def restart(self) -> None:
        self.iterator = iter(self.iterable)

    def resource_name(self):
        return "ChunkColumnMapIterator"

    def size(self) -> int:
        total_size = 0
        for chunk_batch in self.iterator:
            total_size += len(chunk_batch.text)
        self.restart()
        return total_size


class SupervisedColumnMapIterator(data.ColumnMapIterator):
    def __init__(self, iterable: Iterable[SupervisedBatch]):
        data.ColumnMapIterator.__init__(self)

        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def next(self) -> Optional[data.ColumnMap]:
        try:
            batch = next(self.iterator)
            return data.ColumnMap(
                {
                    Mach.TEXT: data.columns.StringColumn(batch.query),
                    Mach.ID: data.columns.TokenColumn(batch.chunk_id),
                }
            )
        except StopIteration:
            return None

    def restart(self):
        self.iterator = iter(self.iterable)

    def resource_name(self):
        return "SupervisedColumnMapIterator"

    def size(self) -> int:
        total_size = 0
        for batch in self.iterator:
            total_size += len(batch.query)
        self.restart()
        return total_size


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
            .text_col(Mach.TEXT)
            .id_col(Mach.ID)
            .tokenizer("char-4")
            .contextual_encoding("none")
            .emb_dim(2000)
            .n_buckets(50000)
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
        train_data = ChunkColumnMapIterator(chunks)

        metrics = kwargs.get("metrics", [])
        if "hash_precision@5" not in metrics:
            metrics.append("hash_precision@5")

        min_epochs, max_epochs = autotune_from_scratch_min_max_epochs(
            size=train_data.size()
        )

        early_stop_callback = EarlyStopWithMinEpochs(
            min_epochs=kwargs.get("epochs", min_epochs),
            tracked_metric=kwargs.get("early_stop_metric", "hash_precision@5"),
            metric_threshold=kwargs.get("early_stop_metric_threshold", 0.95),
        )

        callbacks = [early_stop_callback] + kwargs.get("callbacks", [])

        self.model.coldstart(
            data=train_data,
            strong_cols=[Mach.STRONG],
            weak_cols=[Mach.WEAK],
            learning_rate=kwargs.get("learning_rate", 0.001),
            epochs=kwargs.get("epochs", max_epochs),
            metrics=metrics,
            callbacks=callbacks,
            # max_in_memory_batches=kwargs.get("max_in_memory_batches", None),
            # variable_length=kwargs.get(
            #     "variable_length", data.transformations.VariableLengthConfig()
            # ),
            # batch_size=kwargs.get("batch_size", 2000),
        )

    def supervised_train(self, samples: Iterable[SupervisedBatch], **kwargs):
        train_data = SupervisedColumnMapIterator(samples)

        self.model.train(
            data=train_data,
            learning_rate=kwargs.get("learning_rate", 0.001),
            epochs=kwargs.get("epochs", 3),
            metrics=kwargs.get("metrics", ["hash_precision@5"]),
        )

    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        self.model.erase(ids=chunk_ids)
