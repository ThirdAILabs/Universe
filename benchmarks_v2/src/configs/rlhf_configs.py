import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List
import shutil

import numpy as np
import pandas as pd
import tqdm
from thirdai import bolt
from .cuad_rlhf_preprocessing import process_cuad_data


class RlhfConfig(ABC):
    config_name = None
    dataset_name = None

    @classmethod
    @abstractmethod
    def prepare_data(cls, path_prefix: str):
        pass

    @classmethod
    @abstractmethod
    def pretrain_model(cls, path_prefix: str) -> bolt.UniversalDeepTransformer:
        pass

    @classmethod
    @abstractmethod
    def get_predictions(
        cls, path_prefix: str, model: bolt.UniversalDeepTransformer
    ) -> List[int]:
        pass

    @classmethod
    @abstractmethod
    def get_labels(cls, path_prefix: str) -> List[List[int]]:
        pass

    @classmethod
    @abstractmethod
    def rlhf(cls, path_prefix: str, model: bolt.UniversalDeepTransformer):
        pass

    @classmethod
    @abstractmethod
    def cleanup(cls):
        pass


class WayfairRlhfConfig(RlhfConfig):
    config_name = "wayfair_rlhf"
    dataset = "wayfair"

    unsupervised_data = "wayfair/trn_unsupervised_cleaned.csv"
    test_data = "wayfair/test_queries_match_unsupervised_subsampled.csv"

    @classmethod
    @abstractmethod
    def pretrain_model(cls, path_prefix: str) -> bolt.UniversalDeepTransformer:
        model = bolt.UniversalDeepTransformer(
            data_types={
                "QUERY": bolt.types.text(),
                "PRODUCT_ID": bolt.types.categorical(delimiter=";"),
            },
            target="PRODUCT_ID",
            n_target_classes=931,
            integer_target=True,
            options={
                "extreme_classification": True,
                "extreme_output_dim": 10000,
                "rlhf": True,
            },
        )

        model.cold_start(
            os.path.join(path_prefix, cls.unsupervised_data),
            strong_column_names=["TITLE"],
            weak_column_names=["DESCRIPTION", "BULLET_POINTS", "BRAND"],
            validation=bolt.Validation(
                os.path.join(path_prefix, cls.test_data), ["precision@1"]
            ),
        )

        return model

    @classmethod
    @abstractmethod
    def get_predictions(
        cls, path_prefix: str, model: bolt.UniversalDeepTransformer
    ) -> np.ndarray:
        df = pd.read_csv(os.path.join(path_prefix, cls.test_data))
        batch = [{"QUERY": query} for query in df["QUERY"]]
        preds = model.predict_batch(batch)
        return list(map(lambda x: x[0][0], preds))

    @classmethod
    @abstractmethod
    def get_labels(cls, path_prefix: str) -> List[List[int]]:
        df = pd.read_csv(os.path.join(path_prefix, cls.test_data))
        return [list(map(int, labels.split(";"))) for labels in df["PRODUCT_ID"]]

    @classmethod
    @abstractmethod
    def rlhf(cls, path_prefix: str, model: bolt.UniversalDeepTransformer):
        catalogue_df = pd.read_csv(os.path.join(path_prefix, cls.unsupervised_data))
        labels_to_descriptions = defaultdict(list)

        for _, row in catalogue_df.iterrows():
            labels_to_descriptions[row["PRODUCT_ID"]].append(row["DESCRIPTION"])

        labels_to_descriptions = dict(labels_to_descriptions)

        test_df = pd.read_csv(os.path.join(path_prefix, cls.test_data))
        rlhf_samples = []
        for _, row in test_df.iterrows():
            for product_id in map(int, row["PRODUCT_ID"].split(";")):
                rlhf_samples.append(
                    (
                        {"QUERY": row["QUERY"]},
                        {"QUERY": random.choice(labels_to_descriptions[product_id])},
                    )
                )

        random.shuffle(rlhf_samples)
        batch_size = 2048
        rlhf_batches = [
            rlhf_samples[i : i + batch_size]
            for i in range(0, len(rlhf_samples), batch_size)
        ]

        for batch in tqdm.tqdm(rlhf_batches):
            model.associate(
                batch,
                epochs=1,
                n_buckets=4,
                n_association_samples=1,
                n_balancing_samples=3,
            )


class CuadRlhfConfig(RlhfConfig):
    config_name = "cuad_rlhf"
    dataset_name = "cuad"

    cuad_dataset = "cuad/CUAD_v1/CUAD_v1.json"
    paragraphs_to_answers_filename = "cuad_rlhf/paragraphs_to_answers.csv"
    association_samples_filename = "cuad_rlhf/association_samples.csv"
    questions_and_answers_filename = "cuad_rlhf/questions_and_answers.csv"
    questions_filename = "cuad_rlhf/questions.csv"
    per_contract_data_dirname = "cuad_rlhf/per_contract_data"
    contract_eval_filename = "eval.csv"
    contract_paragraphs_filename = "paragraphs.csv"

    @classmethod
    @abstractmethod
    def prepare_data(cls, path_prefix: str):
        if not os.path.exists("cuad_rlhf"):
            os.makedirs("cuad_rlhf")

        process_cuad_data(
            cuad_dataset=os.path.join(path_prefix, cls.cuad_dataset),
            paragraphs_to_answers_filename=cls.paragraphs_to_answers_filename,
            association_samples_filename=cls.association_samples_filename,
            questions_and_answers_filename=cls.questions_and_answers_filename,
            questions_filename=cls.questions_filename,
            per_contract_data_dirname=cls.per_contract_data_dirname,
            contract_eval_filename=cls.contract_eval_filename,
            contract_paragraphs_filename=cls.contract_paragraphs_filename,
        )

    @classmethod
    @abstractmethod
    def pretrain_model(cls, path_prefix: str) -> bolt.UniversalDeepTransformer:
        model = bolt.UniversalDeepTransformer(
            data_types={
                "text": bolt.types.text(),
                "id": bolt.types.categorical(delimiter=";"),
            },
            target="id",
            n_target_classes=41,
            integer_target=True,
            options={
                "extreme_classification": True,
                "extreme_output_dim": 2000,
                "rlhf": True,
            },
        )

        model.cold_start(
            cls.questions_and_answers_filename,
            strong_column_names=["question"],
            weak_column_names=["answers"],
        )

        model.train(
            cls.paragraphs_to_answers_filename,
            learning_rate=0.0001,
            metrics=["precision@1"],
        )

        return model

    @classmethod
    @abstractmethod
    def get_predictions(
        cls, path_prefix: str, model: bolt.UniversalDeepTransformer
    ) -> List[int]:
        predictions = []
        for contract in tqdm.tqdm(os.listdir(cls.per_contract_data_dirname)):
            model.clear_index()

            paragraphs = os.path.join(
                cls.per_contract_data_dirname,
                contract,
                cls.contract_paragraphs_filename,
            )
            eval_data = os.path.join(
                cls.per_contract_data_dirname, contract, cls.contract_eval_filename
            )

            model.introduce_documents(
                paragraphs,
                strong_column_names=[],
                weak_column_names=["text"],
                verbose=False,
            )

            samples = [{"text": text} for text in pd.read_csv(eval_data)["text"]]
            contract_predictions = model.predict_batch(samples, top_k=1)

            for pred in contract_predictions:
                predictions.append(pred[0][0])

        model.clear_index()

        return predictions

    @classmethod
    @abstractmethod
    def get_labels(cls, path_prefix: str) -> List[List[int]]:
        labels = []
        for contract in tqdm.tqdm(os.listdir(cls.per_contract_data_dirname)):
            eval_data = os.path.join(
                cls.per_contract_data_dirname, contract, cls.contract_eval_filename
            )

            df = pd.read_csv(eval_data)
            if df["id"].dtype == "int64":
                labels.extend([label] for label in df["id"])
            else:
                labels.extend(list(map(int, label.split(";"))) for label in df["id"])

        return labels

    @classmethod
    @abstractmethod
    def rlhf(cls, path_prefix: str, model: bolt.UniversalDeepTransformer):
        association_data = pd.read_csv(cls.association_samples_filename)

        association_samples = []
        for _, row in association_data.iterrows():
            association_samples.append(
                ({"text": row["source"]}, {"text": row["target_answer"]})
            )
        random.shuffle(association_samples)

        batch_size = 2048
        association_batches = [
            association_samples[i : i + batch_size]
            for i in range(0, len(association_samples), batch_size)
        ]

        for batch in association_batches:
            model.associate(
                batch,
                n_buckets=4,
                epochs=1,
                n_association_samples=1,
                n_balancing_samples=3,
            )

    @classmethod
    @abstractmethod
    def cleanup(cls):
        shutil.rmtree("cuad_rlhf")
