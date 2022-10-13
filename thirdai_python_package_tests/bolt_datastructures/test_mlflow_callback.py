import mlflow
import pytest
from test_callbacks import train_model_with_callback
from thirdai.bolt import MlflowCallback

MOCK_TRACKING_URI = "dummy link"
MOCK_EXPERIMENT_NAME = "dummy experiment"
MOCK_RUN_NAME = "dummy run"
MOCK_DATASET_NAME = "dummy dataset"
MOCK_EXPERIMENT_ARGS_KEY = "dummy key"
MOCK_EXPERIMENT_ARGS_VALUE = "dummy value"


@pytest.mark.unit
def test_mlflow_callback(mocker):
    set_tracking_uri_mock = mocker.patch("mlflow.set_tracking_uri")
    set_experiment_mock = mocker.patch("mlflow.set_experiment", return_value=1)
    start_run_mock = mocker.patch("mlflow.start_run")
    instance = start_run_mock.return_value
    instance.info.run_id = 2
    log_param_mock = mocker.patch("mlflow.log_param")
    log_metric_mock = mocker.patch("mlflow.log_metric")

    mlflowcallback = MlflowCallback(
        MOCK_TRACKING_URI,
        MOCK_EXPERIMENT_NAME,
        MOCK_RUN_NAME,
        MOCK_DATASET_NAME,
        {MOCK_EXPERIMENT_ARGS_KEY: MOCK_EXPERIMENT_ARGS_VALUE},
    )

    train_result = train_model_with_callback(mlflowcallback)

    set_tracking_uri_mock.assert_called_once_with(MOCK_TRACKING_URI)
    set_experiment_mock.assert_called_once_with(MOCK_EXPERIMENT_NAME)
    start_run_mock.assert_called_once_with(run_name=MOCK_RUN_NAME)
    log_param_mock.assert_has_calls(
        [
            mocker.call("dataset", MOCK_DATASET_NAME),
            mocker.call(MOCK_EXPERIMENT_ARGS_KEY, MOCK_EXPERIMENT_ARGS_VALUE),
        ]
    )

    train_accuracies = train_result["categorical_accuracy"]
    train_epoch_times = train_result["epoch_times"]

    log_metric_call_params = []
    assert len(train_accuracies) == len(train_epoch_times)
    for acc, time in zip(train_accuracies, train_epoch_times):
        log_metric_call_params.append(("categorical_accuracy", acc))
        log_metric_call_params.append(("epoch_times", time))

    log_metric_mock.assert_has_calls(
        [
            mocker.call(first_param, second_param)
            for first_param, second_param in log_metric_call_params
        ]
    )
