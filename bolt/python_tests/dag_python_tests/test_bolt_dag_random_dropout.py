from ..test_mnist import load_mnist
from .test_bolt_dag_mnist import setup_module, LEARNING_RATE
from thirdai import bolt
import pytest
import time

# Add an integration test marker for all tests in this file
pytestmark = [pytest.mark.integration]

def get_bolt_dag_model(random_dropout=False):

    input_layer = bolt.graph.Input(dim=784)

    hidden_layer = bolt.graph.FullyConnected(
        dim=20000,
        sparsity=0.01,
        activation="relu",
        random_dropout = random_dropout,
        sampling_config=bolt.DWTASamplingConfig(
            num_tables=64, hashes_per_table=3, reservoir_size=32
        ),
    )(input_layer)

    output_layer = bolt.graph.FullyConnected(dim=10, activation="softmax")(hidden_layer)

    

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model

def train_and_predict_model(random_dropout=False):
    model = get_bolt_dag_model(random_dropout)

    train_data, train_labels, test_data, test_labels = load_mnist()

    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=0.0001, epochs=3)
        .silence()
        .with_rebuild_hash_tables(3000)
        .with_reconstruct_hash_functions(10000)
    )

    start_training_time = time.time()
    metrics = model.train(
        train_data=train_data, train_labels=train_labels, train_config=train_config
    )
    end_training_time = time.time()

    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )

    metrics = model.predict(
        test_data=test_data, test_labels=test_labels, predict_config=predict_config
    )

    return metrics[0]["categorical_accuracy"], end_training_time-start_training_time



def test_random_dropout_on_mnist():
    acc_without_random_dropout , train_time_without_random_dropout = train_and_predict_model()
    acc_with_random_dropout , train_time_with_random_dropout = train_and_predict_model(True)

    print('[WITHOUT Random Dropouts]Accuracy ', acc_without_random_dropout, '. Training Time: ', train_time_without_random_dropout)
    print('[WITH Random Dropouts]Accuracy ', acc_with_random_dropout, '. Training Time: ', train_time_with_random_dropout)

