from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from thirdai import bolt
from ..documents import DocumentDataSource
from .models import CancelState, Mach
from ..supervised_datasource import SupDataSource
from ..trainer.training_progress_manager import TrainingProgressManager
from ..utils import clean_text


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
        use_inverted_index: bool,
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
                use_inverted_index=(
                    use_inverted_index if j == 0 else False
                ),  # inverted index will be the same for all models in the ensemble
                mach_index_seed=(mach_index_seed_offset + j * 17),
            )
            for j in range(number_models)
        ]

    @property
    def n_ids(self):
        return self.models[0].n_ids

    def set_mach_sampling_threshodl(self, threshold: float):
        for model in self.models:
            model.set_mach_sampling_threshold(threshold)

    def get_model(self) -> List[bolt.UniversalDeepTransformer]:
        for model in self.models:
            if not model.get_model():
                return None
        return self.models

    def set_model(self, models: List[bolt.UniversalDeepTransformer]):
        for udt_model, ndb_mach in zip(models, self.models):
            ndb_mach.set_model(udt_model)

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass

    def index_documents_impl(
        self,
        training_progress_managers: List[TrainingProgressManager],
        on_progress: Callable,
        cancel_state: CancelState,
    ):
        for progress_manager, model in zip(training_progress_managers, self.models):
            model.index_documents_impl(
                training_progress_manager=progress_manager,
                on_progress=on_progress,
                cancel_state=cancel_state,
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

    def query_mach(self, samples: List, n_results: int, label_probing: bool):
        for model in self.models:
            model.model.set_decode_params(
                min(self.n_ids, n_results), min(self.n_ids, 100)
            )

        # label probing works only when each model has a single hash
        if self.models[0].extreme_num_hashes != 1:
            label_probing = False

        if not label_probing:
            mach_results = bolt.UniversalDeepTransformer.parallel_inference(
                models=[model.model for model in self.models],
                batch=[{self.query_col: clean_text(text)} for text in samples],
            )
            return aggregate_ensemble_results(mach_results)

        else:
            mach_results = bolt.UniversalDeepTransformer.label_probe_mulitple_mach(
                models=[model.model for model in self.models],
                batch=[{self.query_col: clean_text(text)} for text in samples],
            )
            return mach_results

    def query_inverted_index(self, samples, n_results):
        # only the first model in the ensemble can have inverted index
        model = self.models[0]
        if model.inverted_index:
            single_index_results = model.inverted_index.query(
                samples, k=min(n_results, model.n_ids)
            )
            return single_index_results
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
        for model in self.models:
            model.train_on_supervised_data_source(
                supervised_data_source=supervised_data_source,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                max_in_memory_batches=max_in_memory_batches,
                metrics=metrics,
                callbacks=callbacks,
            )
