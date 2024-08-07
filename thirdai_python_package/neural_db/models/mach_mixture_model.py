from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from thirdai import bolt, data

from ..documents import DocumentDataSource
from ..sharded_documents import shard_data_source
from ..supervised_datasource import SupDataSource
from ..trainer.checkpoint_config import (
    CheckpointConfig,
    generate_checkpoint_configs_for_ensembles,
)
from ..trainer.training_progress_manager import TrainingProgressManager
from ..utils import clean_text, pickle_to, requires_condition, unpickle_from
from .mach import Mach
from .model_interface import CancelState, Model, add_retriever_tag, merge_results
from .multi_mach import MultiMach, aggregate_ensemble_results

InferSamples = List
Predictions = Sequence
TrainLabels = List
TrainSamples = List


class MachMixture(Model):
    def __init__(
        self,
        num_shards: int,
        num_models_per_shard: int = 1,
        id_col: str = "DOC_ID",
        id_delimiter: str = " ",
        query_col: str = "QUERY",
        fhr: int = 50_000,
        embedding_dimension: int = 2048,
        extreme_output_dim: int = 10_000,  # for Mach Mixture, we use default dim of 10k
        extreme_num_hashes: int = 8,
        tokenizer="char-4",
        hidden_bias=False,
        model_config=None,
        hybrid=True,
        label_to_segment_map: defaultdict = None,
        seed_for_sharding: int = 0,
        **kwargs,
    ):
        self.id_col = id_col
        self.id_delimiter = id_delimiter
        self.query_col = query_col

        # These parameters are specific to Mach Mixture
        self.num_shards = num_shards
        self.num_models_per_shard = num_models_per_shard

        if label_to_segment_map == None:
            self.label_to_segment_map = defaultdict(list)
        else:
            self.label_to_segment_map = label_to_segment_map

        self.seed_for_sharding = seed_for_sharding

        self.ensembles: List[MultiMach] = [
            MultiMach(
                number_models=num_models_per_shard,
                id_col=id_col,
                id_delimiter=id_delimiter,
                query_col=query_col,
                fhr=fhr,
                embedding_dimension=embedding_dimension,
                extreme_output_dim=extreme_output_dim,
                extreme_num_hashes=extreme_num_hashes,
                tokenizer=tokenizer,
                hidden_bias=hidden_bias,
                hybrid=hybrid,
                model_config=model_config,
                mach_index_seed_offset=j * 341,
            )
            for j in range(self.num_shards)
        ]

    @property
    def shards_data_source(self):
        return self.num_shards > 1

    @property
    def n_ids(self):
        # We assume that the label spaces of underlying ensembles are disjoint (True as of now.)
        n_ids = 0
        for ensemble in self.ensembles:
            n_ids += ensemble.n_ids
        return n_ids

    def set_mach_sampling_threshold(self, threshold: float):
        for ensemble in self.ensembles:
            ensemble.set_mach_sampling_threshold(threshold)

    def get_model(self) -> List[MultiMach]:
        for ensemble in self.ensembles:
            if not ensemble.get_model():
                return None
        return self.ensembles

    def set_model(self, ensembles):
        self.ensembles = ensembles

    def save_meta(self, directory: Path):
        if self.ensembles is not None:
            for i, ensemble in enumerate(self.ensembles):
                ensemble.save_meta(directory / str(i))

        pickle_to(
            [self.label_to_segment_map, self.seed_for_sharding],
            directory / "segment_map_and_seed.pkl",
        )

    def load_meta(self, directory: Path):
        if self.ensembles is not None:
            for i, ensemble in enumerate(self.ensembles):
                ensemble.load_meta(directory / str(i))
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
        training_progress_managers: List[List[TrainingProgressManager]],
        on_progress: Callable,
        cancel_state: CancelState,
        callbacks: List[bolt.train.callbacks.Callback] = None,
    ):
        # This function is the entrypoint to underlying mach models in the mixture. The training progress manager becomes the absolute source of truth in this routine and holds all the data needed to index documents into a model irrespective of whether we are checkpointing or not.
        for progress_manager, ensemble in zip(
            training_progress_managers, self.ensembles
        ):
            ensemble.index_documents_impl(
                training_progress_managers=progress_manager,
                on_progress=on_progress,
                cancel_state=cancel_state,
                callbacks=callbacks,
            )

    def resume(
        self,
        on_progress: Callable,
        cancel_state: CancelState,
        checkpoint_config: CheckpointConfig,
        callbacks: List[bolt.train.callbacks.Callback] = None,
    ):
        # If checkpoint_dir in checkpoint_config is /john/doe and number of models is 2, the underlying mach models will make checkpoint at /john/doe/0 and /john/doe/1 depending on model ids.
        ensemble_checkpoint_configs = generate_checkpoint_configs_for_ensembles(
            config=checkpoint_config,
            number_ensembles=self.num_shards,
            number_models_per_ensemble=self.num_models_per_shard,
        )

        self.load_meta(checkpoint_config.checkpoint_dir)

        # The training manager corresponding to a model loads all the needed to complete the training such as model, document sources, tracker, etc.
        training_managers = []
        for ensemble, config in zip(self.ensembles, ensemble_checkpoint_configs):
            ensemble_training_managers: List[TrainingProgressManager] = []
            for model_id, model in enumerate(ensemble.models):
                # the intro/train shards are only saved for the first model in each ensemble
                if model_id == 0:
                    modelwise_training_manager = (
                        TrainingProgressManager.from_checkpoint(
                            original_mach_model=model,
                            checkpoint_config=config[model_id],
                            for_supervised=False,
                        )
                    )
                else:
                    # for every model other than the first in the ensemble,
                    # manually pass in the loaded intro and train source from
                    # the first model
                    modelwise_training_manager = (
                        TrainingProgressManager.from_checkpoint(
                            original_mach_model=model,
                            checkpoint_config=config[model_id],
                            for_supervised=False,
                            datasource_manager=ensemble_training_managers[
                                0
                            ].datasource_manager,
                        )
                    )
                ensemble_training_managers.append(modelwise_training_manager)
            training_managers.append(ensemble_training_managers)

        self.index_documents_impl(
            training_progress_managers=training_managers,
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
        variable_length: Optional[
            data.transformations.VariableLengthConfig
        ] = data.transformations.VariableLengthConfig(),
        checkpoint_config: CheckpointConfig = None,
        callbacks: List[bolt.train.callbacks.Callback] = None,
        **kwargs,
    ) -> None:
        # We need the original number of classes from the original data source so that we can initialize the Mach models this mixture will have
        number_classes = intro_documents.size

        # Make a sharded data source with introduce documents. When we call shard_data_source, this will shard the introduce data source, return a list of data sources, and modify the label index to keep track of what label goes to what shard
        introduce_data_sources = shard_data_source(
            data_source=intro_documents,
            label_to_segment_map=self.label_to_segment_map,
            number_shards=self.num_shards,
            update_segment_map=True,
        )

        # Once the introduce datasource has been sharded, we can use the update label index to shard the training data source ( We do not want training samples to go to a Mach model that does not contain their labels)
        train_data_sources = shard_data_source(
            train_documents,
            label_to_segment_map=self.label_to_segment_map,
            number_shards=self.num_shards,
            update_segment_map=False,
        )

        # Before we start training individual mach models, we need to save the label to segment map of the current mach mixture so that we can resume in case the training fails.
        if checkpoint_config:
            self.save_meta(checkpoint_config.checkpoint_dir)

        ensemble_checkpoint_configs = generate_checkpoint_configs_for_ensembles(
            config=checkpoint_config,
            number_ensembles=self.num_shards,
            number_models_per_ensemble=self.num_models_per_shard,
        )

        training_managers = []
        for ensemble_id, (intro_shard, train_shard, ensemble, config) in enumerate(
            zip(
                introduce_data_sources,
                train_data_sources,
                self.ensembles,
                ensemble_checkpoint_configs,
            )
        ):
            ensemble_training_managers = []
            for model_id, model in enumerate(ensemble.models):
                modelwise_training_manager = (
                    TrainingProgressManager.from_scratch_for_unsupervised(
                        model=model,
                        intro_documents=intro_shard,
                        train_documents=train_shard,
                        should_train=should_train,
                        fast_approximation=fast_approximation,
                        num_buckets_to_sample=num_buckets_to_sample,
                        max_in_memory_batches=max_in_memory_batches,
                        override_number_classes=number_classes,
                        variable_length=variable_length,
                        checkpoint_config=config[model_id],
                        **kwargs,
                    )
                )
                ensemble_training_managers.append(modelwise_training_manager)
                # When we want to start from scratch, we will have to checkpoint the intro, train sources, the model, tracker,etc. so that the training can be resumed from the checkpoint.
                # only save the intro and train shards for the first model to avoid data duplication. When loading we will load the first and set the intro and train shards for other models in the multimach
                modelwise_training_manager.make_preindexing_checkpoint(
                    save_datasource=model_id == 0
                )  # no-op when checkpoint_config is None.

            training_managers.append(ensemble_training_managers)

        self.index_documents_impl(
            training_progress_managers=training_managers,
            on_progress=on_progress,
            cancel_state=cancel_state,
            callbacks=callbacks,
        )

    def delete_entities(self, entities) -> None:
        if self.shards_data_source:
            segment_to_label_map = defaultdict(list)
            for label in entities:
                segments = self.label_to_segment_map.get(
                    label, []
                )  # Get segments corresponding to the entity
                for segment in segments:
                    segment_to_label_map[segment].append(label)
        else:
            segment_to_label_map = {
                model_id: entities for model_id in range(self.num_shards)
            }

        # Delete entities for each segment
        for i, ensemble in enumerate(self.ensembles):
            ensemble.delete_entities(segment_to_label_map[i])

    def forget_documents(self) -> None:
        for ensemble in self.ensembles:
            ensemble.forget_documents()

    @property
    def searchable(self) -> bool:
        return self.n_ids != 0

    def aggregate_results(self, results, n_results):
        joined_results = []
        for i in range(len(results[0])):
            joined_result = []
            for result in results:
                joined_result.extend(result[i])

            joined_result.sort(key=lambda x: x[1], reverse=True)
            joined_result = joined_result[:n_results]

            joined_results.append(joined_result)
        return joined_results

    def query_mach(self, samples: List, n_results: int, label_probing: bool):
        for ensemble in self.ensembles:
            for model in ensemble.models:
                model.model.set_decode_params(
                    min(model.n_ids, n_results),
                    min(model.n_ids, 100),
                )

        if not label_probing or self.ensembles[0].models[0].extreme_num_hashes != 1:
            ensemble_results = []
            for ensemble in self.ensembles:
                mach_results = bolt.UniversalDeepTransformer.parallel_inference(
                    models=[model.model for model in ensemble.models],
                    batch=[{self.query_col: clean_text(text)} for text in samples],
                )
                ensemble_results.append(aggregate_ensemble_results(mach_results))

        else:
            ensemble_results = (
                bolt.UniversalDeepTransformer.label_probe_multiple_shards(
                    shards=[
                        [model.model for model in ensemble.models]
                        for ensemble in self.ensembles
                    ],
                    batch=[{self.query_col: clean_text(text)} for text in samples],
                )
            )

        return add_retriever_tag(
            self.aggregate_results(ensemble_results, n_results),
            tag="mach",
        )

    def query_finetunable_retriever(self, samples, n_results):
        results = []
        for ensemble in self.ensembles:
            ensemble_result = ensemble.query_finetunable_retriever(samples, n_results)
            if ensemble_result:
                results.append(ensemble_result)

        if not results:
            return None

        return self.aggregate_results(results, n_results)

    def infer_labels(
        self,
        samples: InferSamples,
        n_results: int,
        retriever=None,
        label_probing=True,
        mach_first=False,
        **kwargs,
    ) -> Predictions:
        if not retriever:
            retriever_results = self.query_finetunable_retriever(
                samples, n_results=n_results
            )
            if not retriever_results:
                retriever = "mach"
            else:
                mach_results = self.query_mach(
                    samples, n_results=n_results, label_probing=label_probing
                )
                return [
                    (
                        merge_results(mach_res, retriever_res, n_results)
                        if mach_first
                        # Prioritize retriever_results.
                        else merge_results(retriever_res, mach_res, n_results)
                    )
                    for mach_res, retriever_res in zip(mach_results, retriever_results)
                ]

        if retriever == "mach":
            return self.query_mach(
                samples=samples, n_results=n_results, label_probing=label_probing
            )

        if retriever == "finetunable_retriever":
            results = self.query_finetunable_retriever(
                samples=samples, n_results=n_results
            )
            if not results:
                raise ValueError(
                    "Cannot use retriever 'finetunable_retriever' since the retriever is None."
                )
            return results

        raise ValueError(
            f"Invalid retriever '{retriever}'. Please use 'mach', 'finetunable_retriever', "
            "or pass None to allow the model to autotune which is used."
        )

    def _shard_label_constraints(
        self, entities: List[List[int]]
    ) -> List[List[List[int]]]:
        shards = [[[] for _ in range(len(entities))] for _ in range(self.num_shards)]
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
        if self.shards_data_source:
            sharded_entities = self._shard_label_constraints(entities=entities)
        else:
            sharded_entities = [entities] * self.num_shards

        model_scores = [
            ensemble.score(samples=samples, entities=shard_entity, n_results=n_results)
            for ensemble, shard_entity in zip(self.ensembles, sharded_entities)
        ]

        aggregated_scores = [defaultdict(int) for _ in range(len(samples))]

        for i in range(len(samples)):
            for score in model_scores:
                for label, value, tag in score[i]:
                    aggregated_scores[i][label] += value
                    assert tag == "mach", (
                        "We ignore the retriever tag returned by each ensemble. "
                        "This was inconsequential at the time of writing since "
                        "the MultiMach.score() always returns the 'mach' retriever "
                        "tag. We assert this condition so we reevaluate this "
                        "decision if the condition no longer holds."
                    )

        # Sort the aggregated scores and keep only the top k results
        top_k_results = []
        for i in range(len(samples)):
            sorted_scores = sorted(
                [
                    (label, score, "mach")
                    for label, score in aggregated_scores[i].items()
                ],
                key=lambda x: x[1],
                reverse=True,
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
        for ensemble in self.ensembles:
            ensemble.associate(
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
        shards = [[] for _ in range(self.num_shards)]
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

        for ensemble, shard in zip(self.ensembles, sharded_pairs):
            if len(shard) == 0:
                continue
            ensemble.upvote(
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
            number_shards=self.num_shards,
            label_to_segment_map=self.label_to_segment_map,
            update_segment_map=False,
        )
        for ensemble, shard in zip(self.ensembles, balancing_data_shards):
            ensemble.retrain(
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

    def _resume_supervised(
        self,
        checkpoint_config: Optional[CheckpointConfig],
        callbacks: List[bolt.train.callbacks.Callback],
    ):
        ensemble_checkpoint_configs = generate_checkpoint_configs_for_ensembles(
            config=checkpoint_config,
            number_ensembles=self.num_shards,
            number_models_per_ensemble=self.num_models_per_shard,
        )

        training_managers = []

        for ensemble, config in zip(self.ensembles, ensemble_checkpoint_configs):
            ensemble_training_managers: List[TrainingProgressManager] = []
            for model_id, model in enumerate(ensemble.models):
                if model_id == 0:
                    modelwise_training_manager = (
                        TrainingProgressManager.from_checkpoint(
                            original_mach_model=model,
                            checkpoint_config=config[model_id],
                            for_supervised=True,
                        )
                    )
                else:
                    modelwise_training_manager = (
                        TrainingProgressManager.from_checkpoint(
                            original_mach_model=model,
                            checkpoint_config=config[model_id],
                            for_supervised=True,
                            datasource_manager=ensemble_training_managers[
                                0
                            ].datasource_manager,
                        )
                    )
                ensemble_training_managers.append(modelwise_training_manager)
            training_managers.append(ensemble_training_managers)

        for ensemble, managers in zip(self.ensembles, training_managers):
            ensemble.supervised_training_impl(managers, callbacks=callbacks)

    def _supervised_from_start(
        self,
        supervised_data_source,
        learning_rate,
        epochs,
        batch_size,
        max_in_memory_batches,
        metrics,
        callbacks,
        disable_finetunable_retriever,
        checkpoint_config,
    ):

        supervised_data_source_shards = shard_data_source(
            data_source=supervised_data_source,
            number_shards=self.num_shards,
            label_to_segment_map=self.label_to_segment_map,
            update_segment_map=False,
        )

        ensemble_checkpoint_configs = generate_checkpoint_configs_for_ensembles(
            config=checkpoint_config,
            number_ensembles=self.num_shards,
            number_models_per_ensemble=self.num_models_per_shard,
        )
        training_managers = []

        for ensemble, config, supervised_shard in zip(
            self.ensembles, ensemble_checkpoint_configs, supervised_data_source_shards
        ):
            ensemble_training_managers: List[TrainingProgressManager] = []
            for model_id, model in enumerate(ensemble.models):
                modelwise_training_manager = (
                    TrainingProgressManager.from_scratch_for_supervised(
                        model=model,
                        supervised_datasource=supervised_shard,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        batch_size=batch_size,
                        max_in_memory_batches=max_in_memory_batches,
                        metrics=metrics,
                        disable_finetunable_retriever=disable_finetunable_retriever,
                        checkpoint_config=config[model_id],
                    )
                )
                ensemble_training_managers.append(modelwise_training_manager)
                modelwise_training_manager.make_preindexing_checkpoint(
                    save_datasource=model_id == 0
                )
            training_managers.append(ensemble_training_managers)

        for ensemble, managers in zip(self.ensembles, training_managers):
            ensemble.supervised_training_impl(managers, callbacks=callbacks)

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
            self._supervised_from_start(
                supervised_data_source=supervised_data_source,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                max_in_memory_batches=max_in_memory_batches,
                metrics=metrics,
                callbacks=callbacks,
                disable_finetunable_retriever=disable_finetunable_retriever,
                checkpoint_config=checkpoint_config,
            )
        else:
            self._resume_supervised(
                checkpoint_config=checkpoint_config, callbacks=callbacks
            )
