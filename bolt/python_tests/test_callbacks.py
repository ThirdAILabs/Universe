import pytest
from thirdai import bolt
from utils import gen_numpy_training_data, get_simple_dag_model

pytestmark = [pytest.mark.unit]


N_CLASSES = 10
N_SAMPLES = 1000
BATCH_SIZE = 100
EPOCHS = 10


def train_model_with_callback(callback):
    data, labels = gen_numpy_training_data(
        n_classes=N_CLASSES,
        n_samples=N_SAMPLES,
        noise_std=0.1,
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
        bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=EPOCHS)
        .with_metrics(["categorical_accuracy"])
        .with_callbacks([callback])
    )

    model.train(data, labels, train_config)


class CountCallback(bolt.graph.Callback):
    train_begin_count = 0
    train_end_count = 0
    epoch_begin_count = 0
    epoch_end_count = 0
    batch_begin_count = 0
    batch_end_count = 0

    def on_train_begin(self, model):
        train_begin_count += 1

    def on_train_end(self, model):
        train_end_count += 1

    def on_epoch_begin(self, model):
        epoch_begin_count += 1

    def on_epoch_end(self, model):
        epoch_end_count += 1

    def on_batch_begin(self, model):
        batch_begin_count += 1

    def on_batch_end(self, model):
        batch_end_count += 1


def test_count_callback():
    count_callback = CountCallback()

    train_model_with_callback(count_callback)

    assert count_callback.train_begin_count == 1
