import os
import random

import numpy as np
import pytest
from thirdai import bolt, deployment

pytestmark = [pytest.mark.integration, pytest.mark.release]

CONFIG_FILE = "./saved_oracle_config"

def trained_text_classifier():
    # num_classes, _ = clinc_dataset

    model_config = deployment.ModelConfig(
        input_names=["input"],
        nodes=[
            deployment.FullyConnectedNodeConfig(
                name="hidden",
                dim=deployment.ConstantParameter(1024),
                activation=deployment.ConstantParameter("relu"),
                predecessor="input",
            ),
            deployment.FullyConnectedNodeConfig(
                name="output",
                dim=deployment.UserSpecifiedParameter("output_dim", type=int), # TODO: The value should come from data factory or
                sparsity=deployment.ConstantParameter(1.0),
                activation=deployment.ConstantParameter("softmax"),
                predecessor="hidden",
            ),
        ],
        loss=bolt.CategoricalCrossEntropyLoss(),
    )

    dataset_config = deployment.OracleDatasetFactory(
        config=deployment.UserSpecifiedParameter("config", type=deployment.OracleConfig),
        temporal_context=deployment.UserSpecifiedParameter("temporal_context", type=deployment.TemporalContext, default_value=deployment.TemporalContext.NoneType())
    )

    train_eval_params = deployment.TrainEvalParameters(
        rebuild_hash_tables_interval=None,
        reconstruct_hash_functions_interval=None,
        default_batch_size=2048,
        use_sparse_inference=True,
        evaluation_metrics=["recall@10"],
    )

    config = deployment.DeploymentConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        train_eval_parameters=train_eval_params,
    )

    config.save(CONFIG_FILE)

    context = deployment.TemporalContext()

    model = deployment.ModelPipeline(
        config_path=CONFIG_FILE,
        parameters={
            "output_dim": 3706, 
            "config": deployment.OracleConfig(
                data_types={
                    "userId": bolt.types.categorical(n_unique_classes=6040),
                    "movieId": bolt.types.categorical(n_unique_classes=3706),
                    "timestamp": bolt.types.date()
                },
                temporal_tracking_relationships={
                    "userId": ["movieId"]
                },
                target="movieId"
            ), 
            "context": context
        },
    )

    model.train(
        filename='/Users/benitogeordie/Demos/movielens_train.csv', epochs=3, learning_rate=0.0001
    )

    # return model

trained_text_classifier()