import os

import pandas as pd
import pytest
from distributed_utils import (
    metrics_aggregation_from_workers,
    ray_two_node_cluster_config,
    split_into_2,
)
from thirdai import bolt
from thirdai.demos import (
    download_amazon_kaggle_product_catalog_sampled as download_amazon_kaggle_product_catalog_sampled_wrapped,
)
from thirdai.demos import download_beir_dataset


@pytest.fixture(scope="module")
def download_amazon_kaggle_product_catalog_sampled():
    return download_amazon_kaggle_product_catalog_sampled_wrapped()


@pytest.fixture(scope="module")
def download_scifact_dataset():
    return download_beir_dataset("scifact")


pytestmark = [pytest.mark.distributed]


def download_and_split_catalog_dataset(download_amazon_kaggle_product_catalog_sampled):
    import os

    path = "amazon_product_catalog"
    if not os.path.exists(path):
        os.makedirs(path)

    catalog_file, n_target_classes = download_amazon_kaggle_product_catalog_sampled

    if not os.path.exists(f"{path}/part1") or not os.path.exists(f"{path}/part2"):
        split_into_2(
            file_to_split=catalog_file,
            destination_file_1=f"{path}/part1",
            destination_file_2=f"{path}/part2",
            with_header=True,
        )
    return n_target_classes


def download_and_split_scifact_dataset(download_scifact_dataset):
    import os

    path = "scifact"
    if not os.path.exists(path):
        os.makedirs(path)

    (
        unsupervised_file,
        supervised_trn,
        supervised_tst,
        n_target_classes,
    ) = download_scifact_dataset

    if not os.path.exists(f"{path}/unsupervised_part1") or not os.path.exists(
        f"{path}/unsupervised_part2"
    ):
        split_into_2(
            file_to_split=unsupervised_file,
            destination_file_1=f"{path}/unsupervised_part1",
            destination_file_2=f"{path}/unsupervised_part2",
            with_header=True,
        )

    if not os.path.exists(f"{path}/supervised_trn_part1") or not os.path.exists(
        f"{path}/supervised_trn_part2"
    ):
        split_into_2(
            file_to_split=supervised_trn,
            destination_file_1=f"{path}/supervised_trn_part1",
            destination_file_2=f"{path}/supervised_trn_part2",
            with_header=True,
        )

    return supervised_tst, n_target_classes


def get_udt_scifact_mach_model(n_target_classes):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(delimiter=":"),
        },
        target="DOC_ID",
        n_target_classes=n_target_classes,
        integer_target=True,
        options={"extreme_classification": True, "embedding_dimension": 1024},
    )
    return model


def get_udt_cold_start_model(n_target_classes):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(),
            "PRODUCT_ID": bolt.types.categorical(),
        },
        target="PRODUCT_ID",
        n_target_classes=n_target_classes,
        integer_target=True,
    )
    return model


def test_distributed_mach_cold_start(
    ray_two_node_cluster_config,
    download_scifact_dataset,
):
    supervised_tst, n_target_classes = download_and_split_scifact_dataset(
        download_scifact_dataset
    )

    model = get_udt_scifact_mach_model(n_target_classes)

    metrics = model.cold_start_distributed(
        cluster_config=ray_two_node_cluster_config("linear"),
        filenames=[
            f"{os.getcwd()}/scifact/unsupervised_part1",
            f"{os.getcwd()}/scifact/unsupervised_part2",
        ],
        strong_column_names=["TITLE"],
        weak_column_names=["TEXT"],
        learning_rate=0.001,
        epochs=5,
        metrics=[
            "precision@1",
            "recall@10",
        ],
    )

    overall_metrics = metrics_aggregation_from_workers(metrics["train_metrics"])

    # metrics_aggregation_from_workers just returns metrics for last update
    assert overall_metrics["precision@1"] > 0.90

    validation = bolt.Validation(
        os.path.join(os.getcwd(), supervised_tst),
        metrics=["precision@1"],
        interval=2,
    )

    metrics = model.train_distributed(
        cluster_config=ray_two_node_cluster_config("linear"),
        filenames=[
            f"{os.getcwd()}/scifact/supervised_trn_part1",
            f"{os.getcwd()}/scifact/supervised_trn_part2",
        ],
        learning_rate=0.001,
        epochs=10,
        metrics=[
            "precision@1",
            "recall@10",
        ],
        validation=validation,
    )
    overall_metrics = metrics_aggregation_from_workers(metrics["train_metrics"])

    # metrics_aggregation_from_workers just returns metrics for last update
    assert overall_metrics["precision@1"] > 0.45

    assert metrics["validation_metrics"][-1]["precision@1"] > 0.45


# `ray_two_node_cluster_config` fixture added as parameter to start the mini_cluster
def test_distributed_cold_start(
    ray_two_node_cluster_config, download_amazon_kaggle_product_catalog_sampled
):
    n_target_classes = download_and_split_catalog_dataset(
        download_amazon_kaggle_product_catalog_sampled
    )

    udt_model = get_udt_cold_start_model(n_target_classes)

    metrics = udt_model.cold_start_distributed(
        cluster_config=ray_two_node_cluster_config("linear"),
        filenames=[
            f"{os.getcwd()}/amazon_product_catalog/part1",
            f"{os.getcwd()}/amazon_product_catalog/part2",
        ],
        strong_column_names=["TITLE"],
        weak_column_names=["DESCRIPTION", "BULLET_POINTS", "BRAND"],
        batch_size=2048,
        learning_rate=0.001,
        epochs=5,
        metrics=["categorical_accuracy"],
    )
    overall_metrics = metrics_aggregation_from_workers(metrics["train_metrics"])

    assert overall_metrics["categorical_accuracy"] > 0.7
