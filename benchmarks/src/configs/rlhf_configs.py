import os
import random
import shutil
from abc import ABC
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from thirdai import bolt

from .cuad_rlhf_preprocessing import process_cuad_data


class RlhfConfig(ABC):
    config_name = None
    dataset_name = None

    @classmethod
    def prepare_data(cls, path_prefix: str):
        pass

    @classmethod
    def pretrain_model(cls, path_prefix: str) -> bolt.UniversalDeepTransformer:
        pass

    @classmethod
    def get_predictions(
        cls, path_prefix: str, model: bolt.UniversalDeepTransformer
    ) -> List[int]:
        pass

    @classmethod
    def get_labels(cls, path_prefix: str) -> List[List[int]]:
        pass

    @classmethod
    def rlhf(cls, path_prefix: str, model: bolt.UniversalDeepTransformer):
        pass

    @classmethod
    def cleanup(cls):
        pass


class WayfairRlhfConfig(RlhfConfig):
    config_name = "wayfair_rlhf"
    dataset = "wayfair"

    unsupervised_data = "wayfair/rlhf/trn_unsupervised_cleaned.csv"
    test_data = "wayfair/rlhf/test_queries_cleaned_subsampled.csv"

    @classmethod
    def pretrain_model(cls, path_prefix: str) -> bolt.UniversalDeepTransformer:
        model = bolt.UniversalDeepTransformer(
            data_types={
                "QUERY": bolt.types.text(),
                "PRODUCT_ID": bolt.types.categorical(
                    delimiter=";", n_classes=931, type="int"
                ),
            },
            target="PRODUCT_ID",
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
    def get_predictions(
        cls, path_prefix: str, model: bolt.UniversalDeepTransformer
    ) -> np.ndarray:
        df = pd.read_csv(os.path.join(path_prefix, cls.test_data))
        batch = [{"QUERY": query} for query in df["QUERY"]]
        preds = model.predict_batch(batch)
        return list(map(lambda x: x[0][0], preds))

    @classmethod
    def get_labels(cls, path_prefix: str) -> List[List[int]]:
        df = pd.read_csv(os.path.join(path_prefix, cls.test_data))
        return [list(map(int, labels.split(";"))) for labels in df["PRODUCT_ID"]]

    @classmethod
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
                    (row["QUERY"], random.choice(labels_to_descriptions[product_id]))
                )

        random.shuffle(rlhf_samples)
        batch_size = 2048
        rlhf_batches = [
            rlhf_samples[i : i + batch_size]
            for i in range(0, len(rlhf_samples), batch_size)
        ]

        for batch in rlhf_batches:
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

    preprocessed_data_dir = "cuad_rlhf"
    paragraphs_to_answers_filename = "cuad_rlhf/paragraphs_to_answers.csv"
    association_samples_filename = "cuad_rlhf/association_samples.csv"
    questions_and_answers_filename = "cuad_rlhf/questions_and_answers.csv"
    per_contract_data_dirname = "cuad_rlhf/per_contract_data"
    contract_eval_filename = "eval.csv"
    contract_paragraphs_filename = "paragraphs.csv"

    @classmethod
    def prepare_data(cls, path_prefix: str):
        if os.path.exists(cls.preprocessed_data_dir):
            shutil.rmtree(cls.preprocessed_data_dir)

        os.makedirs(cls.preprocessed_data_dir)

        process_cuad_data(
            cuad_dataset=os.path.join(path_prefix, cls.cuad_dataset),
            paragraphs_to_answers_filename=cls.paragraphs_to_answers_filename,
            association_samples_filename=cls.association_samples_filename,
            questions_and_answers_filename=cls.questions_and_answers_filename,
            per_contract_data_dirname=cls.per_contract_data_dirname,
            contract_eval_filename=cls.contract_eval_filename,
            contract_paragraphs_filename=cls.contract_paragraphs_filename,
        )

    @classmethod
    def pretrain_model(cls, path_prefix: str) -> bolt.UniversalDeepTransformer:
        model = bolt.UniversalDeepTransformer(
            data_types={
                "text": bolt.types.text(),
                "id": bolt.types.categorical(delimiter=";", n_classes=41, type="int"),
            },
            target="id",
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
            epochs=10,
        )

        model.train(
            cls.paragraphs_to_answers_filename, metrics=["precision@1"], epochs=10
        )

        return model

    @classmethod
    def get_predictions(
        cls, path_prefix: str, model: bolt.UniversalDeepTransformer
    ) -> List[int]:
        save_path = "./cuad_pretrained.bolt"
        if os.path.exists(save_path):
            os.remove(save_path)

        # Each contract must be evaluated individually since they share the same
        # questions, so we need to clear the index and introduce contracts one at
        # a time to evalaute. However this will clear the balancing samples and
        # prevent us from doing rlhf after the first evaluation. Thus we save and
        # reload the model to make a copy whose index we can clear for evaluation.
        model.save(save_path)
        model = bolt.UniversalDeepTransformer.load(save_path)

        predictions = []
        for contract in os.listdir(cls.per_contract_data_dirname):
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

        os.remove(save_path)

        return predictions

    @classmethod
    def get_labels(cls, path_prefix: str) -> List[List[int]]:
        labels = []
        for contract in os.listdir(cls.per_contract_data_dirname):
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
    def rlhf(cls, path_prefix: str, model: bolt.UniversalDeepTransformer):
        association_data = pd.read_csv(cls.association_samples_filename)

        association_samples = []
        for _, row in association_data.iterrows():
            association_samples.append((row["source"], row["target_paragraph"]))
        random.shuffle(association_samples)

        batch_size = 512
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
                n_balancing_samples=1,
            )

    @classmethod
    def cleanup(cls):
        shutil.rmtree(cls.preprocessed_data_dir)
