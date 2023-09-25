import json
import os

import pytest
import ray
import thirdai
import thirdai.distributed_bolt as dist
from distributed_utils import setup_ray
from ray import train
from thirdai import bolt, dataset
from thirdai.dataset import RayTextDataSource


@pytest.mark.distributed
def test_ray_file_data_source():
    def training_loop_per_worker(config):
        VOCAB_SIZE = 25
        stream_split_data_iterator = train.get_dataset_shard("train")
        featurizer = dataset.TextGenerationFeaturizer(
            lrc_len=3,
            irc_len=2,
            src_len=1,
            vocab_size=VOCAB_SIZE,
        )
        data_source = RayTextDataSource(stream_split_data_iterator)
        dataset_loader = dataset.DatasetLoader(
            data_source=data_source, featurizer=featurizer, shuffle=True
        )

        data = dataset_loader.load_all(1)
        training_inputs, training_labels = (
            bolt.train.convert_datasets(
                data[:-1], dims=[VOCAB_SIZE, VOCAB_SIZE, (2**32) - 1, VOCAB_SIZE]
            ),
            bolt.train.convert_dataset(data[-1], dim=VOCAB_SIZE),
        )
        assert len(training_inputs) == 8 and len(training_labels) == 8

    data = [
        {"target": "1 2 3 4 5 6"},
        {"target": "7 8 9 10 11 12"},
        {"target": "13 14 15 16 17 18"},
        {"target": "19 20 21 22 23 24"},
    ]
    filename = "output.txt"
    # Write the data to a .txt file
    with open(filename, "w") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")

    train_ray_ds = ray.data.read_text(filename)

    scaling_config = setup_ray()

    # We need to specify `storage_path` in `RunConfig` which must be a networked file system or cloud storage path accessible by all workers. (Ray 2.7.0 onwards)
    run_config = train.RunConfig(storage_path="/share/ray_results")

    trainer = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        scaling_config=scaling_config,
        datasets={"train": train_ray_ds},
        run_config=run_config,
    )

    trainer.fit()
