import random
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from thirdai import bolt, bolt_v2

from .documents import DocumentDataSource
from .utils import clean_text, random_sample

InferSamples = List
Predictions = Sequence
TrainLabels = List
TrainSamples = List


class Model:
    def get_model(self) -> bolt.UniversalDeepTransformer:
        raise NotImplementedError()

    def index_documents(
        self,
        intro_documents: DocumentDataSource,
        train_documents: DocumentDataSource,
        should_train: bool,
        on_progress: Callable = lambda **kwargs: None,
        on_freeze_hash_tables: Callable = lambda **kwargs: None,
    ) -> None:
        raise NotImplementedError()

    def forget_documents(self) -> None:
        raise NotImplementedError()

    def searchable(self) -> bool:
        raise NotImplementedError()

    def get_query_col(self) -> str:
        raise NotImplementedError()

    def get_n_ids(self) -> int:
        raise NotImplementedError()

    def get_id_col(self) -> str:
        raise NotImplementedError()

    def get_id_delimiter(self) -> str:
        raise NotImplementedError()

    def train_samples_to_train_batch(self, samples: TrainSamples):
        query_col = self.get_query_col()
        id_col = self.get_id_col()
        id_delimiter = self.get_id_delimiter()
        return [
            {
                query_col: clean_text(text),
                id_col: id_delimiter.join(map(str, labels)),
            }
            for text, labels in samples
        ]

    def balance_train_label_samples(self, samples: TrainSamples, n_samples: int):
        raise NotImplementedError()

    def balance_train_bucket_samples(self, samples: TrainSamples, n_samples: int):
        raise NotImplementedError()

    def infer_samples_to_infer_batch(self, samples: InferSamples):
        query_col = self.get_query_col()
        return [{query_col: clean_text(text)} for text in samples]

    def train_buckets(
        self, samples: TrainSamples, learning_rate: float, **kwargs
    ) -> None:
        raise NotImplementedError()

    def infer_buckets(
        self, samples: InferSamples, n_results: int, **kwargs
    ) -> Predictions:
        raise NotImplementedError()

    def train_labels(
        self, samples: TrainSamples, learning_rate: float, **kwargs
    ) -> None:
        raise NotImplementedError()

    def infer_labels(
        self, samples: InferSamples, n_results: int, **kwargs
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


def unsupervised_train_on_docs(
    model,
    documents: DocumentDataSource,
    min_epochs: int,
    max_epochs: int,
    metric: str,
    learning_rate: float,
    acc_to_stop: float,
    on_progress: Callable,
    on_freeze_hash_tables: Callable,
):
    model._get_model().freeze_hash_tables()

    for i in range(max_epochs):
        documents.restart()
        metrics = model.cold_start_on_data_source(
            data_source=documents,
            strong_column_names=[documents.strong_column],
            weak_column_names=[documents.weak_column],
            learning_rate=learning_rate,
            epochs=1,
            metrics=[metric],
        )

        val = metrics["train_" + metric][0]
        on_progress(fraction=val)
        if i >= min_epochs - 1 and val > acc_to_stop:
            break


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
        input_dim=50_000,
        hidden_dim=2048,
        extreme_output_dim=50_000,
    ):
        self.id_col = id_col
        self.id_delimiter = id_delimiter
        self.query_col = query_col
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.extreme_output_dim = extreme_output_dim
        self.n_ids = 0
        self.model = None
        self.balancing_samples = []

    def get_model(self) -> bolt.UniversalDeepTransformer:
        return self.model

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass

    def get_n_ids(self) -> int:
        return self.n_ids

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
        on_progress: Callable = lambda **kwargs: None,
        on_freeze_hash_tables: Callable = lambda **kwargs: None,
    ) -> None:
        if intro_documents.id_column != self.id_col:
            raise ValueError(
                f"Model configured to use id_col={self.id_col}, received document with id_col={intro_documents.id_column}"
            )

        if self.model is None:
            self.id_col = intro_documents.id_column
            self.model = self.model_from_scratch(intro_documents)
            learning_rate = 0.005
        else:
            if intro_documents.size() > 0:
                doc_id = intro_documents.id_column
                if doc_id != self.id_col:
                    raise ValueError(
                        f"Document has a different id column ({doc_id}) than the model configuration ({self.id_col})."
                    )
                self.model.introduce_documents_on_data_source(
                    data_source=intro_documents,
                    strong_column_names=[intro_documents.strong_column],
                    weak_column_names=[],
                    num_buckets_to_sample=16,
                )
            learning_rate = 0.001

        self.n_ids += intro_documents.size()
        self.add_balancing_samples(intro_documents)

        if should_train:
            unsupervised_train_on_docs(
                model=self.model,
                documents=train_documents,
                min_epochs=10,
                max_epochs=20,
                metric="hash_precision@5",
                learning_rate=learning_rate,
                acc_to_stop=0.95,
                on_progress=on_progress,
                on_freeze_hash_tables=on_freeze_hash_tables,
            )

    def add_balancing_samples(self, documents: DocumentDataSource):
        samples = make_balancing_samples(documents)
        self.balancing_samples += samples
        if len(self.balancing_samples) > 25000:
            self.balancing_samples = random.sample(self.balancing_samples, k=25000)

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
            n_target_classes=documents.size(),
            integer_target=True,
            options={
                "extreme_classification": True,
                "extreme_output_dim": self.extreme_output_dim,
                "fhr": self.input_dim,
                "embedding_dimension": self.hidden_dim,
                "rlhf": True,
            },
        )

    def forget_documents(self) -> None:
        if self.model is not None:
            self.model.clear_index()
        self.n_ids = 0
        self.balancing_samples = []

    def searchable(self) -> bool:
        return self.n_ids != 0

    def train_labels(
        self, samples: TrainSamples, learning_rate: float, **kwargs
    ) -> None:
        train_batch = self.train_samples_to_train_batch(samples)
        self.model.train_batch(train_batch, learning_rate=learning_rate)

    def infer_labels(
        self, samples: InferSamples, n_results: int, **kwargs
    ) -> Predictions:
        self.model.set_decode_params(min(self.n_ids, n_results), min(self.n_ids, 100))
        infer_batch = self.infer_samples_to_infer_batch(samples)
        all_predictions = self.model.predict_batch(infer_batch)
        #####
        return [
            [int(pred) for pred, _ in predictions] for predictions in all_predictions
        ]

    def train_buckets(self, samples: TrainSamples, learning_rate, **kwargs) -> None:
        train_batch = self.train_samples_to_train_batch(samples)
        self.model.train_with_hashes(train_batch, learning_rate=learning_rate)

    def infer_buckets(
        self, samples: InferSamples, n_results: int, **kwargs
    ) -> Predictions:
        infer_batch = self.infer_samples_to_infer_batch(samples)
        predictions = [
            self.model.predict_hashes(sample)[:n_results] for sample in infer_batch
        ]
        return predictions

    def balance_train_label_samples(self, samples: List, n_samples: int):
        balanced_samples = samples + list(
            random.choices(self.balancing_samples, k=n_samples)
        )
        random.shuffle(balanced_samples)
        return balanced_samples

    def balance_train_bucket_samples(self, samples: List, n_samples: int):
        balancers = random_sample(self.balancing_samples, k=n_samples)
        balancers = [(query, self.get_bucket(labels[0])) for query, labels in balancers]
        balanced_samples = samples + balancers
        random.shuffle(balanced_samples)
        return balanced_samples

    def get_bucket(self, entity: int):
        return self.model.get_index().get_entity_hashes(entity)

    def associate(
        self,
        pairs: List[Tuple[str, str]],
        n_buckets: int,
        n_association_samples: int = 16,
        n_balancing_samples: int = 50,
        learning_rate: float = 0.001,
        epochs: int = 3,
    ):
        query_col = self.get_query_col()

        source_target_samples = [
            (
                {query_col: clean_text(source)},
                {query_col: clean_text(target)},
            )
            for source, target in pairs
        ]

        self.model.associate(
            source_target_samples=source_target_samples,
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
        query_col = self.get_query_col()
        query_col = self.get_query_col()
        samples = [({query_col: clean_text(text)}, label) for text, label in pairs]

        self.model.upvote(
            source_target_samples=samples,
            n_upvote_samples=n_upvote_samples,
            n_balancing_samples=n_balancing_samples,
            learning_rate=learning_rate,
            epochs=epochs,
        )
