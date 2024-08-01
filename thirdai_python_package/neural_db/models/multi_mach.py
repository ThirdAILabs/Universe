from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from thirdai import bolt

from ..documents import DocumentDataSource
from ..supervised_datasource import SupDataSource
from ..trainer.training_progress_manager import TrainingProgressManager
from ..utils import clean_text
from .mach import Mach
from .model_interface import CancelState


def aggregate_ensemble_results(results):
    final_results = []
    for i in range(len(results[0])):
        sample_result = defaultdict(float)
        for model_result in results:
            for res in model_result[i]:
                sample_result[res[0]] += res[1]

        result = [(key, value) for key, value in sample_result.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        final_results.append(result)
    return final_results


class MultiMach:
    def __init__(
        self,
        number_models: int,
        id_col: str,
        id_delimiter: str,
        query_col: str,
        fhr: int,
        embedding_dimension: int,
        extreme_output_dim: int,
        extreme_num_hashes: int,
        tokenizer: int,
        hidden_bias: bool,
        hybrid: bool,
        model_config,
        mach_index_seed_offset: int,
    ):
        if number_models < 1:
            raise ValueError(
                "Cannot initialize a MultiMach with less than one Mach model"
            )
        self.query_col = query_col
        self.models = [
            Mach(
                id_col=id_col,
                id_delimiter=id_delimiter,
                query_col=query_col,
                fhr=fhr,
                embedding_dimension=embedding_dimension,
                extreme_output_dim=extreme_output_dim,
                extreme_num_hashes=extreme_num_hashes,
                tokenizer=tokenizer,
                hidden_bias=hidden_bias,
                model_config=model_config,
                hybrid=(
                    hybrid if j == 0 else False
                ),  # retriever will be the same for all models in the ensemble
                mach_index_seed=(mach_index_seed_offset + j * 17),
            )
            for j in range(number_models)
        ]

    @property
    def n_ids(self):
        return self.models[0].n_ids

    def set_mach_sampling_threshold(self, threshold: float):
        for model in self.models:
            model.set_mach_sampling_threshold(threshold)

    def get_model(self) -> List[bolt.UniversalDeepTransformer]:
        for model in self.models:
            if not model.get_model():
                return None
        return [model.get_model() for model in self.models]

    def set_model(self, models: List[bolt.UniversalDeepTransformer]):
        for udt_model, ndb_mach in zip(models, self.models):
            ndb_mach.set_model(udt_model)

    def save_meta(self, directory: Path):
        for i, model in enumerate(self.models):
            model.save_meta(directory / str(i))

    def load_meta(self, directory: Path):
        for i, model in enumerate(self.models):
            model.load_meta(directory / str(i))

    def index_documents_impl(
        self,
        training_progress_managers: List[TrainingProgressManager],
        on_progress: Callable,
        cancel_state: CancelState,
        callbacks: List[bolt.train.callbacks.Callback] = None,
    ):
        for progress_manager, model in zip(training_progress_managers, self.models):
            model.index_documents_impl(
                training_progress_manager=progress_manager,
                on_progress=on_progress,
                cancel_state=cancel_state,
                callbacks=callbacks,
            )

    def delete_entities(self, entities) -> None:
        for model in self.models:
            model.delete_entities(entities)

    def forget_documents(self) -> None:
        for model in self.models:
            model.forget_documents()

    @property
    def searchable(self) -> bool:
        return self.n_ids != 0

    def query_finetunable_retriever(self, samples, n_results):
        # only the first model in the ensemble can have the retriever
        model = self.models[0]
        if model.finetunable_retriever:
            model.query_finetunable_retriever(samples=samples, n_results=n_results)
        else:
            return None

    def score(self, samples: List, entities: List[List[int]], n_results: int = None):
        model_scores = [
            model.score(samples=samples, entities=entities, n_results=n_results)
            for model in self.models
        ]
        aggregated_scores = [defaultdict(int) for _ in range(len(samples))]

        for i in range(len(samples)):
            for score in model_scores:
                for label, value, tag in score[i]:
                    aggregated_scores[i][label] += value
                    assert tag == "mach", (
                        "We ignore the retriever tag returned by each ensemble. "
                        "This was inconsequential at the time of writing since "
                        "the Mach.score() always returns the 'mach' retriever "
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
        n_association_samples: int,
        n_balancing_samples: int,
        learning_rate: float,
        epochs: int,
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

    def upvote(
        self,
        pairs: List[Tuple[str, int]],
        n_upvote_samples: int,
        n_balancing_samples: int,
        learning_rate: float,
        epochs: int,
    ):
        for model in self.models:
            model.upvote(
                pairs=pairs,
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
        for model in self.models:
            model.retrain(
                balancing_data=balancing_data,
                source_target_pairs=source_target_pairs,
                n_buckets=n_buckets,
                learning_rate=learning_rate,
                epochs=epochs,
            )

    def supervised_training_impl(
        self,
        supervised_progress_managers: List[TrainingProgressManager],
        callbacks: List[bolt.train.callbacks.Callback],
    ):
        for manager, model in zip(supervised_progress_managers, self.models):
            model.supervised_training_impl(manager, callbacks)
