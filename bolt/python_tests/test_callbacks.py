import os.path

import pytest
from thirdai import bolt

from utils import gen_numpy_training_data, get_simple_dag_model

pytestmark = [pytest.mark.unit]


N_CLASSES = 10
N_SAMPLES = 1000
BATCH_SIZE = 100
EPOCHS = 10
SAVE_FILENAME = "fifthEpoch.model"


def train_model_with_callback(callback):
    data, labels = gen_numpy_training_data(
        n_classes=N_CLASSES,
        n_samples=N_SAMPLES,
        noise_std=0.3,
        convert_to_bolt_dataset=True,
        batch_size_for_conversion=BATCH_SIZE,
    )

    model = get_simple_dag_model(
        input_dim=N_CLASSES,
        hidden_layer_dim=2000,
        hidden_layer_sparsity=1.0,
        output_dim=N_CLASSES,
    )

    train_config = (
        bolt.TrainConfig(learning_rate=0.001, epochs=EPOCHS)
        .with_metrics(["categorical_accuracy"])
        .with_callbacks([callback])
    )

    return model.train(data, labels, train_config)


class CountCallback(bolt.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_begin_count = 0
        self.train_end_count = 0
        self.epoch_begin_count = 0
        self.epoch_end_count = 0
        self.batch_begin_count = 0
        self.batch_end_count = 0

    def on_train_begin(self, model, train_state):
        self.train_begin_count += 1

    def on_train_end(self, model, train_state):
        self.train_end_count += 1

    def on_epoch_begin(self, model, train_state):
        self.epoch_begin_count += 1

    def on_epoch_end(self, model, train_state):
        self.epoch_end_count += 1

    def on_batch_begin(self, model, train_state):
        self.batch_begin_count += 1

    def on_batch_end(self, model, train_state):
        self.batch_end_count += 1


def test_callbacks_count_properly():
    count_callback = CountCallback()

    train_model_with_callback(count_callback)

    assert count_callback.train_begin_count == 1
    assert count_callback.train_end_count == 1
    assert count_callback.epoch_begin_count == EPOCHS
    assert count_callback.epoch_end_count == EPOCHS
    assert count_callback.batch_begin_count == (N_SAMPLES / BATCH_SIZE) * EPOCHS
    assert count_callback.batch_end_count == (N_SAMPLES / BATCH_SIZE) * EPOCHS


class SaveOnFifthEpoch(bolt.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_count = 0

    def on_epoch_end(self, model, train_state):
        self.epoch_count += 1
        if self.epoch_count == 5:
            model.save(SAVE_FILENAME)


def test_callbacks_call_cpp_functions():
    save_on_fifth_callback = SaveOnFifthEpoch()

    train_model_with_callback(save_on_fifth_callback)

    assert os.path.isfile(SAVE_FILENAME)

    os.remove(SAVE_FILENAME)


class StopOnFifthEpoch(bolt.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_count = 0

    def on_epoch_end(self, model, train_state):
        self.epoch_count += 1
        if self.epoch_count == 5:
            train_state.stop_training = True


def test_callbacks_stop_correctly():
    stop_on_fifth_callback = StopOnFifthEpoch()

    train_model_with_callback(stop_on_fifth_callback)

    assert stop_on_fifth_callback.epoch_count == 5


class CollectTrainAccuracy(bolt.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.accuracies = []

    def on_train_end(self, model, train_state):
        self.accuracies = train_state.get_train_metric_values("categorical_accuracy")


def test_train_state_correctly_updates_metrics():
    collect_accuracy_callback = CollectTrainAccuracy()

    train_result = train_model_with_callback(collect_accuracy_callback)
    actual_accuracies = train_result["categorical_accuracy"]

    collected_accuracies = collect_accuracy_callback.accuracies

    assert len(actual_accuracies) == len(collected_accuracies)

    for collected_acc, actual_acc in zip(actual_accuracies, collected_accuracies):
        assert collected_acc == actual_acc
