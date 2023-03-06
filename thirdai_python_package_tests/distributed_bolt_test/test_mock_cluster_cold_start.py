import os

import pandas as pd
import pytest
from distributed_utils import ray_two_node_cluster_config, remove_files
from thirdai import bolt

catalog_file = "amazon-kaggle-product-catalog.csv"


def setup_module():
    os.system(
        "curl -L https://www.dropbox.com/s/tf7e5m0cikhcb95/amazon-kaggle-product-catalog-sampled-0.05.csv?dl=0 -o amazon-kaggle-product-catalog.csv"
    )


def get_udt_cold_start_model():
    df = pd.read_csv(f"{os.getcwd()}/{catalog_file}")

    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(),
            "PRODUCT_ID": bolt.types.categorical(),
        },
        target="PRODUCT_ID",
        n_target_classes=df.shape[0],
        integer_target=True,
    )
    return model


# `ray_two_node_cluster_config` fixture added as parameter to start the mini_cluster
def test_distributed_cold_start(ray_two_node_cluster_config):
    udt_model = get_udt_cold_start_model()

    metrics = udt_model.cold_start_distributed(
        cluster_config=ray_two_node_cluster_config("linear"),
        filenames=[f"{os.getcwd()}/xaa", f"{os.getcwd()}/xab"],
        strong_column_names=["TITLE"],
        weak_column_names=["DESCRIPTION", "BULLET_POINTS", "BRAND"],
        batch_size=2048,
        learning_rate=0.001,
        epochs=5,
        metrics=["categorical_accuracy"],
    )
