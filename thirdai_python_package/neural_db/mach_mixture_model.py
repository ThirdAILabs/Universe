from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from thirdai import bolt, data

from .documents import DocumentDataSource
from .models import CancelState, Mach, Model, add_retriever_tag, merge_results
from .sharded_documents import shard_data_source
from .supervised_datasource import SupDataSource
from .trainer.checkpoint_config import (
    CheckpointConfig,
    generate_modelwise_checkpoint_configs,
)
from .trainer.training_progress_manager import TrainingProgressManager
from .utils import clean_text, pickle_to, requires_condition, unpickle_from

InferSamples = List
Predictions = Sequence
TrainLabels = List
TrainSamples = List


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
        extreme_num_hashes: int = 8,
        model_config=None,
        label_to_segment_map: defaultdict = None,
        seed_for_sharding: int = 0,
    ):
        self.id_col = id_col
        self.id_delimiter = id_delimiter
        self.query_col = query_col
        self.fhr = fhr
        self.embedding_dimension = embedding_dimension
        self.extreme_output_dim = extreme_output_dim
        self.extreme_num_hashes = extreme_num_hashes
        self.model_config = model_config

        # These parameters are specific to Mach Mixture
        self.number_models = number_models

        if label_to_segment_map == None:
            self.label_to_segment_map = defaultdict(list)
        else:
            self.label_to_segment_map = label_to_segment_map

        self.seed_for_sharding = seed_for_sharding

        self.models: List[Mach] = [
            Mach(
                id_col=self.id_col,
                id_delimiter=self.id_delimiter,
                query_col=self.query_col,
                fhr=self.fhr,
                embedding_dimension=self.embedding_dimension,
                extreme_output_dim=self.extreme_output_dim,
                extreme_num_hashes=self.extreme_num_hashes,
                model_config=self.model_config,
            )
            for _ in range(self.number_models)
        ]

    @property
    def n_ids(self):
        # We assume that the label spaces of underlying models are disjoint (True as of now.)
        n_ids = 0
        for model in self.models:
            n_ids += model.n_ids
        return n_ids

    def set_mach_sampling_threshold(self, threshold: float):
        if self.models is None:
            raise Exception(
                "Cannot set Sampling Threshold for a model that has not been"
                " initialized"
            )

        for model in self.models:
            model.set_mach_sampling_threshold(threshold)

    def get_model(self) -> List[bolt.UniversalDeepTransformer]:
        for model in self.models:
            if not model.get_model():
                return None
        return self.models

    def set_model(self, models):
        self.models = models

    def save_meta(self, directory: Path):
        if self.models is not None:
            for model in self.models:
                model.save_meta(directory)

        pickle_to(
            [self.label_to_segment_map, self.seed_for_sharding],
            directory / "segment_map_and_seed.pkl",
        )

    def load_meta(self, directory: Path):
        if self.models is not None:
            for model in self.models:
                model.load_meta(directory)
        self.label_to_segment_map, self.seed_for_sharding = unpickle_from(
            directory / "segment_map_and_seed.pkl"
        )

    def get_query_col(self) -> str:
        return self.query_col

    def get_id_col(self) -> str:
        return self.id_col

    def get_id_delimiter(self) -> str:
        return self.id_delimiter

    def index_documents_impl(
        self,
        training_progress_managers: List[TrainingProgressManager],
        on_progress: Callable,
        cancel_state: CancelState,
    ):
        # This function is the entrypoint to underlying mach models in the mixture. The training progress manager becomes the absolute source of truth in this routine and holds all the data needed to index documents into a model irrespective of whether we are checkpointing or not.
        for progress_manager, model in zip(training_progress_managers, self.models):
            model.index_documents_impl(
                training_progress_manager=progress_manager,
                on_progress=on_progress,
                cancel_state=cancel_state,
            )

    def resume(
        self,
        on_progress: Callable,
        cancel_state: CancelState,
        checkpoint_config: CheckpointConfig,
    ):
        # If checkpoint_dir in checkpoint_config is /john/doe and number of models is 2, the underlying mach models will make checkpoint at /john/doe/0 and /john/doe/1 depending on model ids.
        modelwise_checkpoint_configs = generate_modelwise_checkpoint_configs(
            config=checkpoint_config, number_models=self.number_models
        )

        self.load_meta(checkpoint_config.checkpoint_dir)

        # The training manager corresponding to a model loads all the needed to complete the training such as model, document sources, tracker, etc.
        training_managers = []
        for _, (model, config) in enumerate(
            zip(
                self.models,
                modelwise_checkpoint_configs,
            )
        ):
            modelwise_training_manager = TrainingProgressManager.from_checkpoint(
                original_mach_model=model,
                checkpoint_config=config,
            )
            training_managers.append(modelwise_training_manager)

        self.index_documents_impl(
            training_progress_managers=training_managers,
            on_progress=on_progress,
            cancel_state=cancel_state,
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
        variable_length: Optional[
            data.transformations.VariableLengthConfig
        ] = data.transformations.VariableLengthConfig(),
        checkpoint_config: CheckpointConfig = None,
        **kwargs,
    ) -> None:
        # We need the original number of classes from the original data source so that we can initialize the Mach models this mixture will have
        number_classes = intro_documents.size

        # Make a sharded data source with introduce documents. When we call shard_data_source, this will shard the introduce data source, return a list of data sources, and modify the label index to keep track of what label goes to what shard
        introduce_data_sources = shard_data_source(
            data_source=intro_documents,
            label_to_segment_map=self.label_to_segment_map,
            number_shards=self.number_models,
            update_segment_map=True,
        )

        # Once the introduce datasource has been sharded, we can use the update label index to shard the training data source ( We do not want training samples to go to a Mach model that does not contain their labels)
        train_data_sources = shard_data_source(
            train_documents,
            label_to_segment_map=self.label_to_segment_map,
            number_shards=self.number_models,
            update_segment_map=False,
        )

        # Before we start training individual mach models, we need to save the label to segment map of the current mach mixture so that we can resume in case the training fails.
        if checkpoint_config:
            self.save_meta(checkpoint_config.checkpoint_dir)

        modelwise_checkpoint_configs = generate_modelwise_checkpoint_configs(
            config=checkpoint_config, number_models=self.number_models
        )

        training_managers = []
        for _, (intro_shard, train_shard, model, config) in enumerate(
            zip(
                introduce_data_sources,
                train_data_sources,
                self.models,
                modelwise_checkpoint_configs,
            )
        ):
            modelwise_training_manager = TrainingProgressManager.from_scratch(
                model=model,
                intro_documents=intro_shard,
                train_documents=train_shard,
                should_train=should_train,
                fast_approximation=fast_approximation,
                num_buckets_to_sample=num_buckets_to_sample,
                max_in_memory_batches=max_in_memory_batches,
                override_number_classes=number_classes,
                variable_length=variable_length,
                checkpoint_config=config,
                **kwargs,
            )

            training_managers.append(modelwise_training_manager)
            # When we want to start from scratch, we will have to checkpoint the intro, train sources, the model, tracker,etc. so that the training can be resumed from the checkpoint.
            modelwise_training_manager.make_preindexing_checkpoint()  # no-op when checkpoint_config is None.

        self.index_documents_impl(
            training_progress_managers=training_managers,
            on_progress=on_progress,
            cancel_state=cancel_state,
        )

    def delete_entities(self, entities) -> None:
        segment_to_label_map = defaultdict(list)
        for label in entities:
            segments = self.label_to_segment_map.get(
                label, []
            )  # Get segments corresponding to the entity
            for segment in segments:
                segment_to_label_map[segment].append(label)

        # Delete entities for each segment
        for i, model in enumerate(self.models):
            model.delete_entities(segment_to_label_map[i])

    def forget_documents(self) -> None:
        for model in self.models:
            model.forget_documents()

    @property
    def searchable(self) -> bool:
        return self.n_ids != 0

    def aggregate_results(self, results):
        joined_results = []
        for i in range(len(results[0])):
            joined_result = []
            for result in results:
                joined_result.extend(result[i])
            joined_results.append(joined_result)

            joined_result.sort(key=lambda x: x[1], reverse=True)

        return joined_results

    def query_mach(self, samples, n_results):
        for model in self.models:
            model.model.set_decode_params(
                min(self.n_ids, n_results), min(self.n_ids, 100)
            )

        mach_results = bolt.UniversalDeepTransformer.parallel_inference(
            models=[model.model for model in self.models],
            batch=[{self.query_col: clean_text(text)} for text in samples],
        )

        return add_retriever_tag(self.aggregate_results(mach_results), tag="mach")

    def query_inverted_index(self, samples, n_results):
        inverted_index_results = []
        for model in self.models:
            if model.inverted_index:
                single_index_results = model.inverted_index.query(
                    samples, k=min(n_results, model.n_ids)
                )
                inverted_index_results.append(single_index_results)
        if not inverted_index_results:
            return None

        return add_retriever_tag(
            self.aggregate_results(inverted_index_results), tag="inverted_index"
        )

    def infer_labels(
        self,
        samples: InferSamples,
        n_results: int,
        retriever=None,
        **kwargs,
    ) -> Predictions:
        if not retriever:
            index_results = self.query_inverted_index(samples, n_results=n_results)
            if not index_results:
                retriever = "mach"
            else:
                mach_results = self.query_mach(samples, n_results=n_results)
                return [
                    merge_results(mach_res, index_res, n_results)
                    for mach_res, index_res in zip(mach_results, index_results)
                ]

        if retriever == "mach":
            return self.query_mach(samples=samples, n_results=n_results)

        if retriever == "inverted_index":
            results = self.query_inverted_index(samples=samples, n_results=n_results)
            if not results:
                raise ValueError(
                    "Cannot use retriever 'inverted_index' since the index is None."
                )
            return results

        raise ValueError(
            f"Invalid retriever '{retriever}'. Please use 'mach', 'inverted_index', "
            "or pass None to allow the model to autotune which is used."
        )

    def _shard_label_constraints(
        self, entities: List[List[int]]
    ) -> List[List[List[int]]]:
        shards = [[[] for _ in range(len(entities))] for _ in range(self.number_models)]
        for i in range(len(entities)):
            for label in entities[i]:
                model_ids = self.label_to_segment_map.get(label)
                if model_ids is None:
                    raise Exception(f"The Label {label} is not a part of Label Index")
                for model_id in model_ids:
                    shards[model_id][i].append(label)
        return shards

    def score(
        self, samples: InferSamples, entities: List[List[int]], n_results: int = None
    ) -> Predictions:
        sharded_entities = self._shard_label_constraints(entities=entities)

        model_scores = [
            model.score(samples=samples, entities=shard_entity, n_results=n_results)
            for model, shard_entity in zip(self.models, sharded_entities)
        ]

        aggregated_scores = [defaultdict(int) for _ in range(len(samples))]

        for i in range(len(samples)):
            for score in model_scores:
                for label, value, _ in score[i]:
                    aggregated_scores[i][label] += value

        # Sort the aggregated scores and keep only the top k results
        top_k_results = []
        for i in range(len(samples)):
            sorted_scores = sorted(
                aggregated_scores[i].items(), key=lambda x: x[1], reverse=True
            )
            top_k_results.append(
                sorted_scores[:n_results] if n_results else sorted_scores
            )

        return top_k_results

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
        for model in self.models:
            model.associate(
                pairs=pairs,
                n_buckets=n_buckets,
                n_association_samples=n_association_samples,
                n_balancing_samples=n_balancing_samples,
                learning_rate=learning_rate,
                epochs=epochs,
                force_non_empty=kwargs.get("force_non_empty", True),
            )

    def _shard_upvote_pairs(
        self, source_target_pairs: List[Tuple[str, int]]
    ) -> List[List[Tuple[str, int]]]:
        shards = [[] for _ in range(self.number_models)]
        for pair in source_target_pairs:
            model_ids = self.label_to_segment_map.get(pair[1])
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
            if len(shard) == 0:
                continue
            model.upvote(
                pairs=shard,
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
        balancing_data_shards = shard_data_source(
            data_source=balancing_data,
            number_shards=self.number_models,
            label_to_segment_map=self.label_to_segment_map,
            update_segment_map=False,
        )
        for model, shard in zip(self.models, balancing_data_shards):
            model.retrain(
                balancing_data=shard,
                source_target_pairs=source_target_pairs,
                n_buckets=n_buckets,
                learning_rate=learning_rate,
                epochs=epochs,
            )

    def __setstate__(self, state):
        if "model_config" not in state:
            # Add model_config field if an older model is being loaded.
            state["model_config"] = None
        self.__dict__.update(state)

    def train_on_supervised_data_source(
        self,
        supervised_data_source: SupDataSource,
        learning_rate: float,
        epochs: int,
        batch_size: Optional[int],
        max_in_memory_batches: Optional[int],
        metrics: List[str],
        callbacks: List[bolt.train.callbacks.Callback],
    ):
        supervised_data_source_shards = shard_data_source(
            data_source=supervised_data_source,
            number_shards=self.number_models,
            label_to_segment_map=self.label_to_segment_map,
            update_segment_map=False,
        )

        for shard, model in zip(supervised_data_source_shards, self.models):
            if shard.size == 0:
                continue
            model.train_on_supervised_data_source(
                supervised_data_source=shard,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                max_in_memory_batches=max_in_memory_batches,
                metrics=metrics,
                callbacks=callbacks,
            )

    def build_inverted_index(self, documents):
        raise ValueError("This method is not supported on this type of model.")
