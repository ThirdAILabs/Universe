import os
import shutil

import pytest
import ray
import thirdai.distributed_bolt as dist
from distributed_utils import setup_ray
from ray.train.torch import TorchConfig
from thirdai import neural_db

from thirdai_python_package_tests.neural_db.ndb_utils import create_simple_dataset


@pytest.mark.distributed
def test_neural_db_training(create_simple_dataset):
    LOG_PATH = "/tmp/thirdai"
    os.makedirs(LOG_PATH, exist_ok=True)

    filename = create_simple_dataset
    ndb = neural_db.NeuralDB("")

    doc = neural_db.CSV(
        filename,
        id_column="label",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=["text"],
    )

    ndb.insert(sources=[doc], train=False)

    # we are running just one worker since we get OOM issues with multiple workers
    scaling_config = setup_ray(num_workers=1)
    ndb.pretrain_distributed(
        documents=[doc], scaling_config=scaling_config, log_folder=LOG_PATH
    )

    shutil.rmtree(LOG_PATH)
