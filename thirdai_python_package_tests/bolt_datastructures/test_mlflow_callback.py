import pytest
import mlflow
from thirdai.bolt import MlflowCallback
from test_callbacks import train_model_with_callback


MLFLOW_LINK = "http://deplo-mlflo-15qe25sw8psjr-1d20dd0c302edb1f.elb.us-east-1.amazonaws.com"

@pytest.mark.unit
def test_mlflow_callback():
    mlflowcallback = MlflowCallback(
        MLFLOW_LINK,
        "Test Mlflow Callback",
        "test_run_name",
        "test_dataset",
    )

    train_result = train_model_with_callback(mlflowcallback)
    train_accuracies = train_result["categorical_accuracy"]
    train_epoch_times = train_result["epoch_times"]

    client = mlflow.tracking.MlflowClient(MLFLOW_LINK)

    run_id = mlflowcallback.get_run_id()

    accuracy_history = client.get_metric_history(run_id, "categorical_accuracy")
    logged_accuracies = [metric.value for metric in accuracy_history]

    assert len(logged_accuracies) == len(train_accuracies)
    for actual, expected in zip(logged_accuracies, train_accuracies):
        assert actual == expected

    epoch_time_history = client.get_metric_history(run_id, "epoch_times")
    logged_epoch_times = [metric.value for metric in epoch_time_history]

    assert len(logged_epoch_times) == len(train_epoch_times)
    for actual, expected in zip(logged_epoch_times, train_epoch_times):
        assert actual == expected

    client.delete_run(run_id)


