from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import requests
import tqdm
from thirdai import bolt, data, demos, search

from ..documents import DocumentDataSource

# This is unused, we only need this to ensure that old models can be loaded,
# at which point this is removed. The import is here to ensure that on pyinstaller
# builds this file is included in the binary since it is not imported anywhere.
from ..inverted_index import InvertedIndex
from ..supervised_datasource import SupDataSource
from ..trainer.checkpoint_config import CheckpointConfig
from ..trainer.training_progress_manager import (
    TrainingProgressCallback,
    TrainingProgressManager,
)
from ..utils import clean_text, pickle_to
from .finetunable_retriever import FinetunableRetriever
from .mach_defaults import acc_to_stop, metric_to_track
from .model_interface import (
    CancelState,
    InferSamples,
    Model,
    Predictions,
    add_retriever_tag,
    merge_results,
)


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


class ProgressUpdate(bolt.train.callbacks.Callback):
    def __init__(
        self,
        max_epochs,
        progress_callback_fn,
        total_num_batches,
    ):
        super().__init__()

        self.batch_count = 0
        self.max_epochs = max_epochs
        self.progress_callback_fn = progress_callback_fn
        self.total_num_batches = total_num_batches

    def on_batch_end(self):
        self.batch_count += 1

        # We update progress every other batch because otherwise the updates are
        # too fast for frontend components to display these changes.
        if self.batch_count % 2:
            batch_progress = self.batch_count / self.total_num_batches
            progress = batch_progress / self.max_epochs

            # TODO revisit this progress bar update
            # This function (sqrt) increases faster at the beginning
            progress = progress ** (1.0 / 2)
            self.progress_callback_fn(progress)


class FreezeHashTable(bolt.train.callbacks.Callback):
    def __init__(
        self, freeze_before_train, freeze_after_epoch, tracked_metric, metric_threshold
    ):
        super().__init__()

        self.epoch_count = 0
        self.freeze_after_epoch = freeze_after_epoch
        self.tracked_metric = tracked_metric
        self.metric_threshold = metric_threshold
        self.freeze_before_train = freeze_before_train

    def on_train_start(self):
        if self.freeze_before_train:
            self.model.freeze_hash_tables()

    def on_epoch_end(self):
        self.epoch_count += 1
        if self.freeze_before_train:
            return
        if (self.epoch_count == self.freeze_after_epoch) or (
            self.history[f"train_{self.tracked_metric}"][-1] > self.metric_threshold
        ):
            self.model.freeze_hash_tables()


class CancelTraining(bolt.train.callbacks.Callback):
    def __init__(self, cancel_state):
        super().__init__()
        self.cancel_state = cancel_state

    def on_batch_end(self):
        if self.cancel_state is not None and self.cancel_state.is_canceled():
            self.train_state.stop_training()


def download_semantic_enhancement_model(cache_dir, model_name="bolt-splade-medium"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    semantic_model_path = os.path.join(cache_dir, model_name)
    if not os.path.exists(semantic_model_path):
        response = requests.get(
            "https://modelzoo-cdn.azureedge.net/test-models/bolt-splade-medium",
            stream=True,
        )
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 4096  # 4 Kibibyte

        progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(semantic_model_path, "wb") as f:
            for data_chunk in response.iter_content(block_size):
                progress_bar.update(len(data_chunk))
                f.write(data_chunk)
        progress_bar.close()

    vocab_path = os.path.join(cache_dir, "bert-base-uncased.vocab")
    if not os.path.exists(vocab_path):
        demos.bert_base_uncased(dirname=cache_dir)

    return data.transformations.SpladeConfig(
        model_checkpoint=semantic_model_path, tokenizer_vocab=vocab_path
    )


def unsupervised_train_on_docs(
    model,
    documents: DocumentDataSource,
    min_epochs: int,
    max_epochs: int,
    metric: str,
    learning_rate: float,
    batch_size: int,
    acc_to_stop: float,
    on_progress: Callable,
    freeze_before_train: bool,
    freeze_after_epoch: int,
    freeze_after_acc: float,
    cancel_state: CancelState,
    max_in_memory_batches: int,
    variable_length: Optional[data.transformations.VariableLengthConfig],
    training_progress_callback: Optional[TrainingProgressCallback],
    balancing_samples=False,
    semantic_enhancement=False,
    semantic_model_cache_dir=".cache/neural_db_semantic_model",
    coldstart_callbacks: List[bolt.train.callbacks.Callback] = None,
    **kwargs,
):
    documents.restart()

    early_stop_callback = EarlyStopWithMinEpochs(
        min_epochs=min_epochs, tracked_metric=metric, metric_threshold=acc_to_stop
    )

    progress_callback = ProgressUpdate(
        max_epochs=max_epochs,
        progress_callback_fn=on_progress,
        total_num_batches=(
            math.ceil(documents.size / batch_size)
            if batch_size
            else math.ceil(documents.size / 2048)  # default batch size we use in UDT.
        ),
    )

    cancel_training_callback = CancelTraining(cancel_state=cancel_state)

    freeze_hashtable_callback = FreezeHashTable(
        freeze_before_train=freeze_before_train,
        freeze_after_epoch=freeze_after_epoch,
        tracked_metric=metric,
        metric_threshold=freeze_after_acc,
    )

    callbacks = [
        early_stop_callback,
        progress_callback,
        cancel_training_callback,
        freeze_hashtable_callback,
    ]

    if coldstart_callbacks:
        callbacks.extend(coldstart_callbacks)

    if training_progress_callback:
        callbacks.append(training_progress_callback)

    splade_config = None
    if semantic_enhancement:
        splade_config = download_semantic_enhancement_model(semantic_model_cache_dir)

    if balancing_samples:
        model.cold_start_with_balancing_samples(
            data=documents,
            strong_column_names=[documents.strong_column],
            weak_column_names=[documents.weak_column],
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=max_epochs,
            train_metrics=[metric],
            callbacks=callbacks,
            variable_length=variable_length,
        )
    else:
        model.cold_start_on_data_source(
            data_source=documents,
            strong_column_names=[documents.strong_column],
            weak_column_names=[documents.weak_column],
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=max_epochs,
            metrics=[metric],
            callbacks=callbacks,
            max_in_memory_batches=max_in_memory_batches,
            variable_length=variable_length,
            splade_config=splade_config,
        )


def make_balancing_samples(documents: DocumentDataSource):
    samples = [
        (". ".join([row.strong, row.weak]), [row.id])
        for row in documents.row_iterator()
    ]
    if len(samples) > 25000:
        samples = random.sample(samples, k=25000)
    return samples


class Mach(Model):
    def __init__(
        self,
        id_col="DOC_ID",
        id_delimiter=" ",
        query_col="QUERY",
        fhr=50_000,
        embedding_dimension=2048,
        extreme_output_dim=50_000,
        extreme_num_hashes=8,
        tokenizer="char-4",
        hidden_bias=False,
        model_config=None,
        hybrid=True,
        mach_index_seed: int = 341,
        index_max_shard_size=8_000_000,
        **kwargs,
    ):
        self.id_col = id_col
        self.id_delimiter = id_delimiter
        self.tokenizer = tokenizer
        self.query_col = query_col
        self.fhr = fhr
        self.embedding_dimension = embedding_dimension
        self.extreme_output_dim = extreme_output_dim
        self.extreme_num_hashes = extreme_num_hashes
        self.hidden_bias = hidden_bias
        self.n_ids = 0
        self.model = None
        self.balancing_samples = []
        self.model_config = model_config
        self.mach_index_seed = mach_index_seed

        if hybrid:
            self.finetunable_retriever = FinetunableRetriever()
        else:
            self.finetunable_retriever = None

    def set_mach_sampling_threshold(self, threshold: float):
        if self.model is None:
            raise Exception(
                "Cannot set Sampling Threshold for a model that has not been"
                " initialized"
            )
        self.model.set_mach_sampling_threshold(threshold)

    def reset_model(self, new_model: Mach):
        self.id_col = new_model.id_col
        self.id_delimiter = new_model.id_delimiter
        self.tokenizer = new_model.tokenizer
        self.query_col = new_model.query_col
        self.fhr = new_model.fhr
        self.embedding_dimension = new_model.embedding_dimension
        self.extreme_output_dim = new_model.extreme_output_dim
        self.extreme_num_hashes = new_model.extreme_num_hashes
        self.hidden_bias = new_model.hidden_bias
        self.n_ids = new_model.n_ids
        self.model = new_model.model
        self.balancing_samples = new_model.balancing_samples
        self.model_config = new_model.model_config
        self.finetunable_retriever = new_model.finetunable_retriever

    def save(self, path: Path):
        pickle_to(self, filepath=path)
        self.save_meta(path.parent / "model")

    def get_model(self) -> bolt.UniversalDeepTransformer:
        return self.model

    def set_model(self, model):
        self.model = model

    def save_meta(self, directory: Path):
        if self.finetunable_retriever:
            self.finetunable_retriever.save_meta(directory / "finetunable_retriever")

    def load_meta(self, directory: Path):
        if self.finetunable_retriever:
            self.finetunable_retriever.load_meta(directory / "finetunable_retriever")

    def set_n_ids(self, n_ids: int):
        self.n_ids = n_ids

    def get_query_col(self) -> str:
        return self.query_col

    def get_id_col(self) -> str:
        return self.id_col

    def get_id_delimiter(self) -> str:
        return self.id_delimiter

    def introduce_documents(
        self,
        intro_documents: DocumentDataSource,
        fast_approximation: bool,
        num_buckets_to_sample: Optional[int],
        override_number_classes: int,
    ):
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

        if self.finetunable_retriever:
            intro_documents.restart()
            self.finetunable_retriever.index_from_start(intro_documents)

        self.n_ids += intro_documents.size

    def index_documents_impl(
        self,
        training_progress_manager: TrainingProgressManager,
        on_progress: Callable = lambda **kwargs: None,
        cancel_state: CancelState = None,
        callbacks: List[bolt.train.callbacks.Callback] = None,
    ):
        intro_documents = training_progress_manager.intro_source
        train_documents = training_progress_manager.train_source

        if not training_progress_manager.is_insert_completed:
            self.introduce_documents(
                intro_documents=intro_documents,
                **training_progress_manager.introduce_arguments(),
            )
            training_progress_manager.insert_complete()

        if not training_progress_manager.is_training_completed:
            train_arguments = training_progress_manager.training_arguments()
            unsupervised_train_on_docs(
                model=self.model,
                documents=train_documents,
                metric=metric_to_track,
                acc_to_stop=acc_to_stop,
                on_progress=on_progress,
                cancel_state=cancel_state,
                training_progress_callback=TrainingProgressCallback(
                    training_progress_manager=training_progress_manager
                ),
                coldstart_callbacks=callbacks,
                **train_arguments,
            )
            training_progress_manager.training_complete()

    def resume(
        self,
        on_progress: Callable,
        cancel_state: CancelState,
        checkpoint_config: CheckpointConfig,
        callbacks: List[bolt.train.callbacks.Callback] = None,
    ):
        # This will load the datasources, model, training config and upload the current model with the loaded one. This updates the underlying UDT MACH of the current model with the one from the checkpoint along with other class attributes.
        training_progress_manager = TrainingProgressManager.from_checkpoint(
            self, checkpoint_config=checkpoint_config, for_supervised=False
        )

        self.index_documents_impl(
            training_progress_manager=training_progress_manager,
            on_progress=on_progress,
            cancel_state=cancel_state,
            callbacks=callbacks,
        )

    def index_from_start(
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
        variable_length: Optional[
            data.transformations.VariableLengthConfig
        ] = data.transformations.VariableLengthConfig(),
        checkpoint_config: CheckpointConfig = None,
        callbacks: List[bolt.train.callbacks.Callback] = None,
        **kwargs,
    ):
        """
        override_number_classes : The number of classes for the Mach model

        Note: Given the datasources for introduction and training, we initialize a Mach model that has number_classes set to the size of introduce documents. But if we want to use this Mach model in our mixture of Models, this will not work because each Mach will be initialized with number of classes equal to the size of the datasource shard. Hence, we add override_number_classes parameters which if set, will initialize Mach Model with number of classes passed by the Mach Mixture.
        """

        training_progress_manager = (
            TrainingProgressManager.from_scratch_for_unsupervised(
                model=self,
                intro_documents=intro_documents,
                train_documents=train_documents,
                should_train=should_train,
                fast_approximation=fast_approximation,
                num_buckets_to_sample=num_buckets_to_sample,
                max_in_memory_batches=max_in_memory_batches,
                override_number_classes=override_number_classes,
                variable_length=variable_length,
                checkpoint_config=checkpoint_config,
                **kwargs,
            )
        )

        training_progress_manager.make_preindexing_checkpoint()
        self.index_documents_impl(
            training_progress_manager=training_progress_manager,
            on_progress=on_progress,
            cancel_state=cancel_state,
            callbacks=callbacks,
        )

    def add_balancing_samples(self, documents: DocumentDataSource):
        samples = make_balancing_samples(documents)
        self.balancing_samples += samples
        if len(self.balancing_samples) > 25000:
            self.balancing_samples = random.sample(self.balancing_samples, k=25000)

    def delete_entities(self, entities) -> None:
        for entity in entities:
            self.get_model().forget(entity)

        if self.finetunable_retriever:
            self.finetunable_retriever.delete_entities(entities)

    def model_from_scratch(
        self, documents: DocumentDataSource, number_classes: int = None
    ):
        model = bolt.UniversalDeepTransformer(
            data_types={
                self.query_col: bolt.types.text(tokenizer=self.tokenizer),
                self.id_col: bolt.types.categorical(
                    n_classes=(
                        documents.size if number_classes is None else number_classes
                    ),
                    delimiter=self.id_delimiter,
                    type="int",
                ),
            },
            target=self.id_col,
            extreme_classification=True,
            extreme_output_dim=self.extreme_output_dim,
            fhr=self.fhr,
            embedding_dimension=self.embedding_dimension,
            extreme_num_hashes=self.extreme_num_hashes,
            hidden_bias=self.hidden_bias,
            rlhf=True,
            mach_index_seed=self.mach_index_seed,
            model_config=self.model_config,
        )
        model.insert_new_doc_ids(documents)
        return model

    def forget_documents(self) -> None:
        if self.model is not None:
            self.model.clear_index()
        self.n_ids = 0
        self.balancing_samples = []

        if self.finetunable_retriever:
            self.finetunable_retriever.forget_documents()

    @property
    def searchable(self) -> bool:
        return self.n_ids != 0

    def query_mach(self, samples, n_results):
        self.model.set_decode_params(min(self.n_ids, n_results), min(self.n_ids, 100))
        infer_batch = self.infer_samples_to_infer_batch(samples)
        return add_retriever_tag(
            results=self.model.predict_batch(infer_batch), tag="mach"
        )

    def query_finetunable_retriever(self, samples, n_results):
        return self.finetunable_retriever.infer_labels(
            samples=samples, n_results=min(self.n_ids, n_results)
        )

    def infer_labels(
        self,
        samples: InferSamples,
        n_results: int,
        retriever=None,
        mach_first=False,
        **kwargs,
    ) -> Predictions:
        if not retriever:
            if not self.finetunable_retriever:
                retriever = "mach"
            else:
                mach_results = self.query_mach(samples=samples, n_results=n_results)
                index_results = self.query_finetunable_retriever(
                    samples=samples, n_results=n_results
                )
                return [
                    (
                        merge_results(mach_res, index_res, n_results)
                        if mach_first
                        # Prioritize retriver results.
                        else merge_results(index_res, mach_res, n_results)
                    )
                    for mach_res, index_res in zip(mach_results, index_results)
                ]

        if retriever == "mach":
            return self.query_mach(samples=samples, n_results=n_results)

        if retriever == "finetunable_retriever":
            if not self.finetunable_retriever:
                raise ValueError(
                    "Cannot use retriever 'finetunable_retriever' since the retriever is None."
                )
            return self.query_finetunable_retriever(
                samples=samples, n_results=n_results
            )

        raise ValueError(
            f"Invalid retriever '{retriever}'. Please use 'mach', 'finetunable_retriever', "
            "or pass None to allow the model to autotune which is used."
        )

    def score(
        self, samples: InferSamples, entities: List[List[int]], n_results: int = None
    ) -> Predictions:
        infer_batch = self.infer_samples_to_infer_batch(samples)
        results = self.model.score_batch(infer_batch, classes=entities, top_k=n_results)
        return add_retriever_tag(results=results, tag="mach")

    def _format_associate_samples(self, pairs: List[Tuple[str, str]]):
        return [(clean_text(source), clean_text(target)) for source, target in pairs]

    def associate(
        self,
        pairs: List[Tuple[str, str]],
        n_buckets: int,
        n_association_samples: int = 16,
        n_balancing_samples: int = 50,
        learning_rate: float = 0.001,
        epochs: int = 3,
        **kwargs,
    ):
        self.model.associate(
            source_target_samples=self._format_associate_samples(pairs),
            n_buckets=n_buckets,
            n_association_samples=n_association_samples,
            n_balancing_samples=n_balancing_samples,
            learning_rate=learning_rate,
            epochs=epochs,
            force_non_empty=kwargs.get("force_non_empty", True),
        )

        if self.finetunable_retriever:
            self.finetunable_retriever.associate(pairs)

    def upvote(
        self,
        pairs: List[Tuple[str, int]],
        n_upvote_samples: int = 16,
        n_balancing_samples: int = 50,
        learning_rate: float = 0.001,
        epochs: int = 3,
    ):
        samples = [(clean_text(text), label) for text, label in pairs]

        self.model.upvote(
            source_target_samples=samples,
            n_upvote_samples=n_upvote_samples,
            n_balancing_samples=n_balancing_samples,
            learning_rate=learning_rate,
            epochs=epochs,
        )

        if self.finetunable_retriever:
            self.finetunable_retriever.upvote(pairs)

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
        if "finetunable_retriever" not in state:
            state["finetunable_retriever"] = None
        if "inverted_index" in state:
            state["finetunable_retriever"] = FinetunableRetriever(
                search.FinetunableRetriever.train_from(state["inverted_index"].export())
            )
        self.__dict__.update(state)

    def supervised_training_impl(
        self,
        supervised_progress_manager: TrainingProgressManager,
        callbacks: List[bolt.train.callbacks.Callback],
    ):
        if not supervised_progress_manager.is_training_completed:
            train_args = supervised_progress_manager.training_arguments()
            self.model.train_on_data_source(
                data_source=supervised_progress_manager.train_source,
                callbacks=callbacks
                + [
                    TrainingProgressCallback(
                        training_progress_manager=supervised_progress_manager
                    )
                ],
                **train_args,
            )

            if (
                supervised_progress_manager.tracker._train_state.disable_finetunable_retriever
            ):
                self.finetunable_retriever = None
            elif self.finetunable_retriever:
                supervised_progress_manager.train_source.restart()
                self.finetunable_retriever.train_on_supervised_data_source(
                    supervised_progress_manager.train_source
                )

            supervised_progress_manager.training_complete()

    def train_on_supervised_data_source(
        self,
        supervised_data_source: SupDataSource,
        learning_rate: float,
        epochs: int,
        batch_size: Optional[int],
        max_in_memory_batches: Optional[int],
        metrics: List[str],
        callbacks: List[bolt.train.callbacks.Callback],
        disable_finetunable_retriever: bool,
        checkpoint_config: Optional[CheckpointConfig] = None,
    ):
        if (
            checkpoint_config is None
            or checkpoint_config.resume_from_checkpoint is False
        ):
            training_manager = TrainingProgressManager.from_scratch_for_supervised(
                model=self,
                supervised_datasource=supervised_data_source,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                max_in_memory_batches=max_in_memory_batches,
                metrics=metrics,
                disable_finetunable_retriever=disable_finetunable_retriever,
                checkpoint_config=checkpoint_config,
            )
            training_manager.make_preindexing_checkpoint(save_datasource=True)
        else:
            training_manager = TrainingProgressManager.from_checkpoint(
                self, checkpoint_config, for_supervised=True
            )

        self.supervised_training_impl(training_manager, callbacks=callbacks)
