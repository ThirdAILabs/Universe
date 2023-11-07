import math
import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from thirdai import bolt

from .documents import DocumentDataSource
from .sharded_documents import ShardedDataSource
from .utils import clean_text, random_sample

InferSamples = List
Predictions = Sequence
TrainLabels = List
TrainSamples = List


# This class can be constructed by clients that use neural_db.
# The object can then be passed into Model.index_documents(), and if
# the client calls CancelState.cancel() on the object, training will halt.
class CancelState:
    def __init__(self, canceled=False):
        self.canceled = canceled

    def cancel(self):
        self.canceled = True

    def uncancel(self):
        self.canceled = False

    def is_canceled(self):
        return self.canceled


class Model:
    def get_model(self) -> bolt.UniversalDeepTransformer:
        raise NotImplementedError()

    def index_documents(
        self,
        intro_documents: DocumentDataSource,
        train_documents: DocumentDataSource,
        should_train: bool,
        fast_approximation: bool = True,
        num_buckets_to_sample: Optional[int] = None,
        on_progress: Callable = lambda **kwargs: None,
        cancel_state: CancelState = None,
        max_in_memory_batches: int = None,
    ) -> None:
        raise NotImplementedError()

    def forget_documents(self) -> None:
        raise NotImplementedError()

    def delete_entities(self, entities) -> None:
        raise NotImplementedError()

    @property
    def searchable(self) -> bool:
        raise NotImplementedError()

    def get_query_col(self) -> str:
        raise NotImplementedError()

    def set_n_ids(self, n_ids: int):
        raise NotImplementedError()

    def get_id_col(self) -> str:
        raise NotImplementedError()

    def get_id_delimiter(self) -> str:
        raise NotImplementedError()

    def infer_samples_to_infer_batch(self, samples: InferSamples):
        query_col = self.get_query_col()
        return [{query_col: clean_text(text)} for text in samples]

    def infer_buckets(
        self, samples: InferSamples, n_results: int, **kwargs
    ) -> Predictions:
        raise NotImplementedError()

    def infer_labels(
        self,
        samples: InferSamples,
        n_results: int,
        **kwargs,
    ) -> Predictions:
        raise NotImplementedError()

    def score(
        self, samples: InferSamples, entities: List[List[int]], n_results: int = None
    ) -> Predictions:
        raise NotImplementedError()

    def save_meta(self, directory: Path) -> None:
        raise NotImplementedError()

    def load_meta(self, directory: Path):
        raise NotImplementedError()

    def associate(
        self,
        pairs: List[Tuple[str, str]],
        n_buckets: int,
        n_association_samples: int = 16,
        n_balancing_samples: int = 50,
        learning_rate: float = 0.001,
        epochs: int = 3,
    ):
        raise NotImplementedError()

    def upvote(
        self,
        pairs: List[Tuple[str, int]],
        n_upvote_samples: int = 16,
        n_balancing_samples: int = 50,
        learning_rate: float = 0.001,
        epochs: int = 3,
    ):
        raise NotImplementedError()

    def retrain(
        self,
        balancing_data: DocumentDataSource,
        source_target_pairs: List[Tuple[str, str]],
        n_buckets: int,
        learning_rate: float,
        epochs: int,
    ):
        raise NotImplementedError()


class EarlyStopWithMinEpochs(bolt.train.callbacks.Callback):
    def __init__(
        self,
        min_epochs,
        tracked_metric,
        metric_threshold,
    ):
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


class ProgressUpdate(bolt.train.callbacks.Callback):
    def __init__(
        self,
        max_epochs,
        progress_callback_fn,
    ):
        super().__init__()

        self.batch_count = 0
        self.max_epochs = max_epochs
        self.progress_callback_fn = progress_callback_fn

    def on_batch_end(self):
        self.batch_count += 1

        # We update progress every other epoch because otherwise the updates are
        # too fast for frontend components to display these changes.
        if self.batch_count % 2:
            batch_progress = self.batch_count / self.train_state.batches_in_dataset()
            progress = batch_progress / self.max_epochs

            # TODO revisit this progress bar update
            # This function (sqrt) increases faster at the beginning
            progress = progress ** (1.0 / 2)
            self.progress_callback_fn(progress)


class CancelTraining(bolt.train.callbacks.Callback):
    def __init__(self, cancel_state):
        super().__init__()
        self.cancel_state = cancel_state

    def on_batch_end(self):
        if self.cancel_state is not None and self.cancel_state.is_canceled():
            self.train_state.stop_training()


def unsupervised_train_on_docs(
    model,
    documents: DocumentDataSource,
    min_epochs: int,
    max_epochs: int,
    metric: str,
    learning_rate: float,
    acc_to_stop: float,
    on_progress: Callable,
    freeze_before_train: bool,
    cancel_state: CancelState,
    max_in_memory_batches: int,
):
    if freeze_before_train:
        model._get_model().freeze_hash_tables()

    documents.restart()

    early_stop_callback = EarlyStopWithMinEpochs(
        min_epochs=min_epochs,
        tracked_metric=metric,
        metric_threshold=acc_to_stop,
    )

    progress_callback = ProgressUpdate(
        max_epochs=max_epochs,
        progress_callback_fn=on_progress,
    )

    cancel_training_callback = CancelTraining(cancel_state=cancel_state)

    model.cold_start_on_data_source(
        data_source=documents,
        strong_column_names=[documents.strong_column],
        weak_column_names=[documents.weak_column],
        learning_rate=learning_rate,
        epochs=max_epochs,
        metrics=[metric],
        callbacks=[early_stop_callback, progress_callback, cancel_training_callback],
        max_in_memory_batches=max_in_memory_batches,
    )


def make_balancing_samples(documents: DocumentDataSource):
    samples = [
        (". ".join([row.strong, row.weak]), [row.id])
        for row in documents.row_iterator()
    ]
    if len(samples) > 25000:
        samples = random.sample(samples, k=25000)
    return samples


def autotune_from_scratch_min_max_epochs(size):
    if size < 1000:
        return 10, 15
    if size < 10000:
        return 8, 13
    if size < 100000:
        return 5, 10
    if size < 1000000:
        return 3, 8
    return 1, 5


def autotune_from_base_min_max_epochs(size):
    if size < 100000:
        return 5, 10
    if size < 1000000:
        return 3, 8
    return 1, 5


class Mach(Model):
    def __init__(
        self,
        id_col="DOC_ID",
        id_delimiter=" ",
        query_col="QUERY",
        fhr=50_000,
        embedding_dimension=2048,
        extreme_output_dim=50_000,
        model_config=None,
    ):
        self.id_col = id_col
        self.id_delimiter = id_delimiter
        self.query_col = query_col
        self.fhr = fhr
        self.embedding_dimension = embedding_dimension
        self.extreme_output_dim = extreme_output_dim
        self.n_ids = 0
        self.model = None
        self.balancing_samples = []
        self.model_config = model_config

    def set_mach_sampling_threshold(self, threshold: float):
        if self.model is None:
            raise Exception(
                "Cannot set Sampling Threshold for a model that has not been"
                " initialized"
            )
        self.model.set_mach_sampling_threshold(threshold)

    def get_model(self) -> bolt.UniversalDeepTransformer:
        return self.model

    def set_model(self, model):
        self.model = model

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass

    def set_n_ids(self, n_ids: int):
        self.n_ids = n_ids

    def get_query_col(self) -> str:
        return self.query_col

    def get_id_col(self) -> str:
        return self.id_col

    def get_id_delimiter(self) -> str:
        return self.id_delimiter

    def index_documents(
        self,
        intro_documents: DocumentDataSource,
        train_documents: DocumentDataSource,
        should_train: bool,
        fast_approximation: bool = True,
        num_buckets_to_sample: Optional[int] = None,
        on_progress: Callable = lambda **kwargs: None,
        cancel_state: CancelState = None,
        max_in_memory_batches: int = None,
        override_number_classes: int = None,
    ) -> None:
        """
        override_number_classes : The number of classes for the Mach model

        Note: Given the datasources for introduction and training, we initialize a Mach model that has number_classes set to the size of introduce documents. But if we want to use this Mach model in our mixture of Models, this will not work because each Mach will be initialized with number of classes equal to the size of the datasource shard. Hence, we add override_number_classes parameters which if set, will initialize Mach Model with number of classes passed by the Mach Mixture.
        """
        if intro_documents.id_column != self.id_col:
            raise ValueError(
                f"Model configured to use id_col={self.id_col}, received document with"
                f" id_col={intro_documents.id_column}"
            )

        if self.model is None:
            self.id_col = intro_documents.id_column
            self.model = self.model_from_scratch(
                intro_documents, number_classes=override_number_classes
            )
            learning_rate = 0.005
            freeze_before_train = False
            min_epochs, max_epochs = autotune_from_scratch_min_max_epochs(
                train_documents.size
            )
        else:
            if intro_documents.size > 0:
                doc_id = intro_documents.id_column
                if doc_id != self.id_col:
                    raise ValueError(
                        f"Document has a different id column ({doc_id}) than the model"
                        f" configuration ({self.id_col})."
                    )

                num_buckets_to_sample = num_buckets_to_sample or int(
                    self.model.get_index().num_hashes() * 2.0
                )

                self.model.introduce_documents_on_data_source(
                    data_source=intro_documents,
                    strong_column_names=[intro_documents.strong_column],
                    weak_column_names=[intro_documents.weak_column],
                    fast_approximation=fast_approximation,
                    num_buckets_to_sample=num_buckets_to_sample,
                )
            learning_rate = 0.001
            # Freezing at the beginning prevents the model from forgetting
            # things it learned from pretraining.
            freeze_before_train = True
            # Less epochs here since it converges faster when trained on a base
            # model.
            min_epochs, max_epochs = autotune_from_base_min_max_epochs(
                train_documents.size
            )

        self.n_ids += intro_documents.size
        self.add_balancing_samples(intro_documents)

        if should_train and train_documents.size > 0:
            unsupervised_train_on_docs(
                model=self.model,
                documents=train_documents,
                min_epochs=min_epochs,
                max_epochs=max_epochs,
                metric="hash_precision@5",
                learning_rate=learning_rate,
                acc_to_stop=0.95,
                on_progress=on_progress,
                freeze_before_train=freeze_before_train,
                cancel_state=cancel_state,
                max_in_memory_batches=max_in_memory_batches,
            )

    def add_balancing_samples(self, documents: DocumentDataSource):
        samples = make_balancing_samples(documents)
        self.balancing_samples += samples
        if len(self.balancing_samples) > 25000:
            self.balancing_samples = random.sample(self.balancing_samples, k=25000)

    def delete_entities(self, entities) -> None:
        for entity in entities:
            self.get_model().forget(entity)

    def model_from_scratch(
        self, documents: DocumentDataSource, number_classes: int = None
    ):
        return bolt.UniversalDeepTransformer(
            data_types={
                self.query_col: bolt.types.text(tokenizer="char-4"),
                self.id_col: bolt.types.categorical(delimiter=self.id_delimiter),
            },
            target=self.id_col,
            n_target_classes=(
                documents.size if number_classes is None else number_classes
            ),
            integer_target=True,
            options={
                "extreme_classification": True,
                "extreme_output_dim": self.extreme_output_dim,
                "fhr": self.fhr,
                "embedding_dimension": self.embedding_dimension,
                "rlhf": True,
            },
            model_config=self.model_config,
        )

    def forget_documents(self) -> None:
        if self.model is not None:
            self.model.clear_index()
        self.n_ids = 0
        self.balancing_samples = []

    @property
    def searchable(self) -> bool:
        return self.n_ids != 0

    def infer_labels(
        self,
        samples: InferSamples,
        n_results: int,
        **kwargs,
    ) -> Predictions:
        infer_batch = self.infer_samples_to_infer_batch(samples)
        self.model.set_decode_params(min(self.n_ids, n_results), min(self.n_ids, 100))
        return self.model.predict_batch(infer_batch)

    def score(
        self, samples: InferSamples, entities: List[List[int]], n_results: int = None
    ) -> Predictions:
        infer_batch = self.infer_samples_to_infer_batch(samples)
        return self.model.score_batch(infer_batch, classes=entities, top_k=n_results)

    def infer_buckets(
        self, samples: InferSamples, n_results: int, **kwargs
    ) -> Predictions:
        infer_batch = self.infer_samples_to_infer_batch(samples)
        predictions = [
            self.model.predict_hashes(sample)[:n_results] for sample in infer_batch
        ]
        return predictions

    def _format_associate_samples(self, pairs: List[Tuple[str, str]]):
        query_col = self.get_query_col()

        return [
            ({query_col: clean_text(source)}, {query_col: clean_text(target)})
            for source, target in pairs
        ]

    def associate(
        self,
        pairs: List[Tuple[str, str]],
        n_buckets: int,
        n_association_samples: int = 16,
        n_balancing_samples: int = 50,
        learning_rate: float = 0.001,
        epochs: int = 3,
    ):
        self.model.associate(
            source_target_samples=self._format_associate_samples(pairs),
            n_buckets=n_buckets,
            n_association_samples=n_association_samples,
            n_balancing_samples=n_balancing_samples,
            learning_rate=learning_rate,
            epochs=epochs,
        )

    def upvote(
        self,
        pairs: List[Tuple[str, int]],
        n_upvote_samples: int = 16,
        n_balancing_samples: int = 50,
        learning_rate: float = 0.001,
        epochs: int = 3,
    ):
        samples = [
            ({self.get_query_col(): clean_text(text)}, label) for text, label in pairs
        ]

        self.model.upvote(
            source_target_samples=samples,
            n_upvote_samples=n_upvote_samples,
            n_balancing_samples=n_balancing_samples,
            learning_rate=learning_rate,
            epochs=epochs,
        )

    def retrain(
        self,
        balancing_data: DocumentDataSource,
        source_target_pairs: List[Tuple[str, str]],
        n_buckets: int,
        learning_rate: float,
        epochs: int,
    ):
        self.model.associate_cold_start_data_source(
            balancing_data=balancing_data,
            strong_column_names=[balancing_data.strong_column],
            weak_column_names=[balancing_data.weak_column],
            source_target_samples=self._format_associate_samples(source_target_pairs),
            n_buckets=n_buckets,
            n_association_samples=1,
            learning_rate=learning_rate,
            epochs=epochs,
            metrics=["hash_precision@5"],
            options=bolt.TrainOptions(),
        )

    def __setstate__(self, state):
        if "model_config" not in state:
            # Add model_config field if an older model is being loaded.
            state["model_config"] = None
        self.__dict__.update(state)


class MachMixture(Model):
    def __init__(
        self,
        number_models: int,
        id_col: str = "DOC_ID",
        id_delimiter: str = " ",
        query_col: str = "QUERY",
        fhr: int = 50_000,
        embedding_dimension: int = 2048,
        extreme_output_dim: int = 10_000,  # for Mach Mixture, we use default dim of 10k
        model_config=None,
        label_index: dict = {},
        seed_for_sharding: int = 0,
    ):
        print("Initializing a Mixture of Mach Models")
        self.number_models = number_models
        self.id_col = id_col
        self.id_delimiter = id_delimiter
        self.query_col = query_col
        self.fhr = fhr
        self.embedding_dimension = embedding_dimension
        self.extreme_output_dim = extreme_output_dim
        self.n_ids = 0
        self.models = None
        self.balancing_samples = []
        self.model_config = model_config
        self.label_index = label_index
        self.seed_for_sharding = seed_for_sharding

    def set_mach_sampling_threshold(self, threshold: float):
        if self.models is None:
            raise Exception(
                "Cannot set Sampling Threshold for a model that has not been"
                " initialized"
            )

        for model in self.models:
            model.set_mach_sampling_threshold(threshold)

    def get_model(self) -> List[bolt.UniversalDeepTransformer]:
        return self.models

    def set_model(self, models):
        self.models = models

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass

    def set_n_ids(self, n_ids: int):
        self.n_ids = n_ids

    def get_query_col(self) -> str:
        return self.query_col

    def get_id_col(self) -> str:
        return self.id_col

    def get_id_delimiter(self) -> str:
        return self.id_delimiter

    def index_documents(
        self,
        intro_documents: DocumentDataSource,
        train_documents: DocumentDataSource,
        should_train: bool,
        fast_approximation: bool = True,
        num_buckets_to_sample: Optional[int] = None,
        on_progress: Callable = lambda **kwargs: None,
        cancel_state: CancelState = None,
        max_in_memory_batches: int = None,
    ) -> None:
        if intro_documents.id_column != self.id_col:
            raise ValueError(
                f"Model configured to use id_col={self.id_col}, received document with"
                f" id_col={intro_documents.id_column}"
            )

        print("Made the Sharded Data Source")
        sharded_data_source = ShardedDataSource(
            document_data_source=intro_documents,
            number_shards=self.number_models,
            label_index=self.label_index,
            seed=self.seed_for_sharding,
        )
        # Start Sharding the dataset
        print("Begin Sharding the Introduce dataset")
        introduce_data_sources = sharded_data_source.shard_data_source()

        if not self.models:
            self.id_col = intro_documents.id_column
            self.models = [
                self.model_from_scratch(intro_documents)
                for _ in range(self.number_models)
            ]
            learning_rate = 0.005
            min_epochs, max_epochs = autotune_from_scratch_min_max_epochs(
                int(train_documents.size / self.number_models)
            )

        else:
            if intro_documents.size > 0:
                doc_id = intro_documents.id_column
                if doc_id != self.id_col:
                    raise ValueError(
                        f"Document has a different id column ({doc_id}) than the model"
                        f" configuration ({self.id_col})."
                    )

                num_buckets_to_sample = num_buckets_to_sample or int(
                    self.models[0].get_index().num_hashes() * 2.0
                )

                for model, data_source_shard in zip(
                    self.models, introduce_data_sources
                ):
                    model.introduce_documents_on_data_source(
                        data_source=data_source_shard,
                        strong_column_names=[data_source_shard.strong_column],
                        weak_column_names=[data_source_shard.weak_column],
                        fast_approximation=fast_approximation,
                        num_buckets_to_sample=num_buckets_to_sample,
                    )
            learning_rate = 0.001
            # Freezing at the beginning prevents the model from forgetting
            # things it learned from pretraining.
            freeze_before_train = True
            # Less epochs here since it converges faster when trained on a base
            # model.
            min_epochs, max_epochs = autotune_from_base_min_max_epochs(
                int(train_documents.size / self.number_models)
            )

        self.n_ids += intro_documents.size
        self.add_balancing_samples(intro_documents)

        train_data_sources = sharded_data_source.shard_using_index(
            train_documents,
            label_index=self.label_index,
            number_shards=self.number_models,
        )

        if should_train:
            for model, train_data_source_shard in zip(self.models, train_data_sources):
                unsupervised_train_on_docs(
                    model=self.model,
                    documents=train_data_source_shard,
                    min_epochs=min_epochs,
                    max_epochs=max_epochs,
                    metric="hash_precision@5",
                    learning_rate=learning_rate,
                    acc_to_stop=0.95,
                    on_progress=on_progress,
                    freeze_before_train=freeze_before_train,
                    cancel_state=cancel_state,
                    max_in_memory_batches=max_in_memory_batches,
                )

    def add_balancing_samples(self, documents: DocumentDataSource):
        samples = make_balancing_samples(documents)
        self.balancing_samples += samples
        if len(self.balancing_samples) > 25000:
            self.balancing_samples = random.sample(self.balancing_samples, k=25000)

    def delete_entities(self, entities) -> None:
        for model in self.models:
            for entity in entities:
                model.forget(entity)

    def model_from_scratch(
        self,
        documents: DocumentDataSource,
    ):
        return bolt.UniversalDeepTransformer(
            data_types={
                self.query_col: bolt.types.text(tokenizer="char-4"),
                self.id_col: bolt.types.categorical(delimiter=self.id_delimiter),
            },
            target=self.id_col,
            n_target_classes=documents.size,
            integer_target=True,
            options={
                "extreme_classification": True,
                "extreme_output_dim": self.extreme_output_dim,
                "fhr": self.fhr,
                "embedding_dimension": self.embedding_dimension,
                "rlhf": True,
            },
            model_config=self.model_config,
        )

    def forget_documents(self) -> None:
        if self.models:
            for model in self.models:
                model.clear_index()
        self.n_ids = 0
        self.balancing_samples = []

    @property
    def searchable(self) -> bool:
        return self.n_ids != 0

    def infer_labels(
        self,
        samples: InferSamples,
        n_results: int,
        **kwargs,
    ) -> Predictions:
        infer_batch = self.infer_samples_to_infer_batch(samples)
        results = [[] for _ in range(len(samples))]
        for model in self.models:
            model.set_decode_params(min(self.n_ids, n_results), min(self.n_ids, 100))
            single_model_outputs = model.predict_batch(infer_batch)
            for index in range(len(samples)):
                results[index].extend(single_model_outputs[index])

        for index in range(len(results)):
            results[index].sort(key=lambda x: x[1], reverse=True)
            results[index] = results[index][:n_results]
        return results

    def score(
        self, samples: InferSamples, entities: List[List[int]], n_results: int = None
    ) -> Predictions:
        raise NotImplementedError()

    def infer_buckets(
        self, samples: InferSamples, n_results: int, **kwargs
    ) -> Predictions:
        NotImplementedError()

    def _format_associate_samples(self, pairs: List[Tuple[str, str]]):
        query_col = self.get_query_col()

        return [
            ({query_col: clean_text(source)}, {query_col: clean_text(target)})
            for source, target in pairs
        ]

    def associate(
        self,
        pairs: List[Tuple[str, str]],
        n_buckets: int,
        n_association_samples: int = 16,
        n_balancing_samples: int = 50,
        learning_rate: float = 0.001,
        epochs: int = 3,
    ):
        for model in self.models:
            model.associate(
                source_target_samples=self._format_associate_samples(pairs),
                n_buckets=n_buckets,
                n_association_samples=n_association_samples,
                n_balancing_samples=n_balancing_samples,
                learning_rate=learning_rate,
                epochs=epochs,
            )

    def _shard_upvote_pairs(self, source_target_pairs: List[Tuple[str, int]]):
        shards = [[] for _ in range(self.number_models)]
        for pair in source_target_pairs:
            model_ids = self.label_index.get(pair[1])
            if model_ids is None:
                raise Exception(f"The Label {pair[1]} is not a part of Label Index")
            for model_id in model_ids:
                shards[model_id].append(pair)
        return shards

    def upvote(
        self,
        pairs: List[Tuple[str, int]],
        n_upvote_samples: int = 16,
        n_balancing_samples: int = 50,
        learning_rate: float = 0.001,
        epochs: int = 3,
    ):
        sharded_pairs = self._shard_upvote_pairs(pairs)

        for model, shard in zip(self.models, sharded_pairs):
            samples = [
                ({self.get_query_col(): clean_text(text)}, label)
                for text, label in shard
            ]
            model.upvote(
                source_target_samples=samples,
                n_upvote_samples=n_upvote_samples,
                n_balancing_samples=n_balancing_samples,
                learning_rate=learning_rate,
                epochs=epochs,
            )

    def retrain(
        self,
        balancing_data: DocumentDataSource,
        source_target_pairs: List[Tuple[str, str]],
        n_buckets: int,
        learning_rate: float,
        epochs: int,
    ):
        sharded_data_source = ShardedDataSource(
            document_data_source=balancing_data,
            number_shards=self.number_models,
            label_index=self.label_index,
            seed=self.seed_for_sharding,
        )
        balancing_data_shards = sharded_data_source.shard_data_source()

        for model, shard in zip(self.models, balancing_data_shards):
            model.associate_cold_start_data_source(
                balancing_data=shard,
                strong_column_names=[balancing_data.strong_column],
                weak_column_names=[balancing_data.weak_column],
                source_target_samples=self._format_associate_samples(
                    source_target_pairs
                ),
                n_buckets=n_buckets,
                n_association_samples=1,
                learning_rate=learning_rate,
                epochs=epochs,
                metrics=["hash_precision@5"],
                options=bolt.TrainOptions(),
            )

    def __setstate__(self, state):
        if "model_config" not in state:
            # Add model_config field if an older model is being loaded.
            state["model_config"] = None
        self.__dict__.update(state)
