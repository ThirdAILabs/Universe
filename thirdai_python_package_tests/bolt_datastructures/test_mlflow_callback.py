import pytest
from thirdai import bolt, dataset
from test_callbacks import train_model_with_callback


@pytest.mark.unit
def test_mlflow_callback():
    mlflowcallback = bolt.MlflowCallback(
        "http://deplo-mlflo-15qe25sw8psjr-1d20dd0c302edb1f.elb.us-east-1.amazonaws.com",
        "test_mlflow_experiment",
        "test_run_name",
        "test_dataset",
    )

    train_model_with_callback(mlflowcallback)
