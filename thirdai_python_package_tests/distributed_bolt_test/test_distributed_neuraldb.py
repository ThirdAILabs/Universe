import os
import shutil

import pytest
import ray
import thirdai.distributed_bolt as dist
from distributed_utils import setup_ray
from ray.air import session
from ray.train.torch import TorchConfig
from thirdai import neural_db

from thirdai_python_package_tests.neural_db.ndb_utils import create_simple_dataset


@pytest.mark.unit
def test_neural_db_training(create_simple_dataset):
    filename = create_simple_dataset
    ndb = neural_db.NeuralDB("")

    doc = neural_db.CSV(
        filename,
        id_column="id",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=[],
    )

    ndb.insert(sources=[doc], train=False)

    scaling_config = setup_ray()

    ndb.train_distributed(documents=[doc], scaling_config=scaling_config)
