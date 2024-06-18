from typing import Dict, Iterable, List, Optional, Set, Tuple

from thirdai import bolt, data
from thirdai.neural_db.models.mach_defaults import autotune_from_scratch_min_max_epochs

from ..core.retriever import Retriever
from ..core.types import ChunkBatch, ChunkId, Score, SupervisedBatch


class ChunkColumnMapIterator(data.ColumnMapIterator):
    def __init__(
        self,
        iterable: Iterable[ChunkBatch],
        text_columns: Dict[str, str],
        multi_label=False,
    ):
        data.ColumnMapIterator.__init__(self)

        self.iterable = iterable
        self.iterator = iter(self.iterable)
        self.text_columns = text_columns
        self.multi_label = multi_label

    def next(self) -> Optional[data.ColumnMap]:
        id_column = (
            data.columns.TokenArrayColumn
            if self.multi_label
            else data.columns.TokenColumn
        )

        try:
            batch = next(self.iterator)
            columns = {
                Mach.ID: id_column(batch.chunk_id.to_list(), dim=data.columns.MAX_DIM)
            }
            for name, attr in self.text_columns.items():
                columns[name] = data.columns.StringColumn(getattr(batch, attr))
            return data.ColumnMap(columns)
        except StopIteration:
            return None

    def restart(self) -> None:
        self.iterator = iter(self.iterable)

    def resource_name(self):
        return "ChunkColumnMapIterator"

    def size(self) -> int:
        return sum(len(batch.text) for batch in self.iterable)


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

    def __init__(
        self,
        tokenizer: str = "char-4",
        encoding: str = "none",
        emb_dim: int = 2000,
        n_buckets: int = 50000,
        output_act: str = "sigmoid",
        emb_bias: bool = True,
        output_bias: bool = True,
        n_hashes: Optional[int] = None,
        index_seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        config = (
            bolt.MachConfig()
            .text_col(Mach.TEXT)
            .id_col(Mach.ID)
            .tokenizer(tokenizer)
            .contextual_encoding(encoding)
            .emb_dim(emb_dim)
            .n_buckets(n_buckets)
            .output_activation(output_act)
            .emb_bias(emb_bias)
            .output_bias(output_bias)
        )
        if index_seed:
            config = config.index_seed(index_seed)
        if n_hashes:
            config = config.n_hashes(n_hashes)

        self.model = config.build()

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
        choices: List[Set[ChunkId]],
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
        train_data = ChunkColumnMapIterator(
            samples, text_columns={Mach.TEXT: "query"}, multi_label=True
        )

        self.model.train(
            data=train_data,
            learning_rate=learning_rate,
            epochs=epochs,
            metrics=metrics or ["hash_precision@5"],
        )

    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        self.model.erase(ids=chunk_ids)

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str):
        instance = cls()
        instance.model = bolt.MachRetriever.load(path)
        return instance
