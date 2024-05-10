from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from thirdai import bolt, data

from ..documents import DocumentDataSource
from ..supervised_datasource import SupDataSource
from ..trainer.checkpoint_config import CheckpointConfig
from ..utils import clean_text

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
        override_number_classes: int = None,
        variable_length: Optional[
            data.transformations.VariableLengthConfig
        ] = data.transformations.VariableLengthConfig(),
        checkpoint_config: CheckpointConfig = None,
        **kwargs,
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

    def infer_labels(
        self,
        samples: InferSamples,
        n_results: int,
        retriever: Optional[str] = None,
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
        **kwargs,
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
    ):
        raise NotImplementedError()


def normalize_scores(results):
    if len(results) == 0:
        return results
    if len(results) == 1:
        return [(results[0][0], 1.0, results[0][2])]
    ids, scores, retriever = zip(*results)
    scores = np.array(scores)
    scores -= np.min(scores)
    scores /= np.max(scores)
    return list(zip(ids, scores, retriever))


def merge_results(results_a, results_b, k):
    results_a = normalize_scores(results_a)
    results_b = normalize_scores(results_b)
    results = []
    cache = set()

    min_len = min(len(results_a), len(results_b))
    for a, b in zip(results_a, results_b):
        if a[0] not in cache:
            results.append(a)
            cache.add(a[0])
        if b[0] not in cache:
            results.append(b)
            cache.add(b[0])

    if len(results) < k:
        for i in range(min_len, len(results_a)):
            if results_a[i][0] not in cache:
                results.append(results_a[i])
        for i in range(min_len, len(results_b)):
            if results_b[i][0] not in cache:
                results.append(results_b[i])

    return results[:k]


def add_retriever_tag(results, tag):
    return [[(id, score, tag) for id, score in result] for result in results]
