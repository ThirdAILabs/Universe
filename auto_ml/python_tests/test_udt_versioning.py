import os
import random
from collections import defaultdict

import pandas as pd
import pytest
from download_dataset_fixtures import download_amazon_kaggle_product_catalog_sampled
from model_test_utils import compute_evaluate_accuracy
from thirdai import bolt

pytestmark = [pytest.mark.unit]


def test_udt_load_old_version(download_amazon_kaggle_product_catalog_sampled):
    catalog_file, n_target_classes = download_amazon_kaggle_product_catalog_sampled

    model = bolt.UniversalDeepTransformer(
        data_types={
            "Sample": bolt.types.text(),
            "Target": bolt.types.categorical(),
        },
        target="Target",
        n_target_classes=10,
    )

    


    assert final_metric.ending_train_metric > 0.5
    assert metrics["categorical_accuracy"][-1] == final_metric.ending_train_metric
