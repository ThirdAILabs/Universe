from hashlib import new
import numpy as np
from thirdai import bolt, dataset
import pytest
import time
import os
import shutil

# Returns data and labels for learning the function f(a) = a, where a is
# sparse (num_true_labels_per_example number of nonzeros).
def generate_random_easy_sparse(output_dim, num_true_labels_per_example, num_examples):
    label_indices = np.random.choice(
        output_dim, size=(num_examples * num_true_labels_per_example)
    )
    label_values = np.ones(shape=label_indices.shape) / num_true_labels_per_example
    label_offsets = np.arange(0, len(label_values) + 1, num_true_labels_per_example)
    data_indices = label_indices
    data_values = np.ones(shape=data_indices.shape)
    data_offsets = label_offsets
    return (
        data_indices.astype("uint32"),
        data_values.astype("float32"),
        data_offsets.astype("uint32"),
    ), (
        label_indices.astype("uint32"),
        label_values.astype("float32"),
        label_offsets.astype("uint32"),
    )


def build_and_train_mach(
    num_train=10000,
    num_true_labels_per_sample=10,
    input_and_output_dim=1000,
    learning_rate=0.001,
    batch_size=512,
    num_epochs=5,
):

    train_x_np, train_y_np = generate_random_easy_sparse(
        output_dim=input_and_output_dim,
        num_true_labels_per_example=num_true_labels_per_sample,
        num_examples=num_train,
    )

    mach = bolt.Mach(
        max_label=input_and_output_dim,
        num_classifiers=4,
        input_dim=input_and_output_dim,
        hidden_layer_dim=input_and_output_dim,
        hidden_layer_sparsity=1,
        last_layer_dim=input_and_output_dim // 10,
        last_layer_sparsity=1,
        use_softmax=True,
    )

    mach.train(
        train_x_np=train_x_np,
        train_y_np=train_y_np,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )

    return mach


def get_recall(result, test_y, num_true_labels_per_sample):
    count = 0
    for i, start in enumerate(range(0, len(test_y[0]), num_true_labels_per_sample)):
        end = start + num_true_labels_per_sample
        if result[i] in test_y[0][start:end]:
            count += 1
    recall = count / (len(test_y[2]) - 1)
    # print("Recall: ", recall)
    return recall


@pytest.mark.unit
def test_mlflow_callback():
    mlflowcallback = bolt.MlflowCallback("test_mlflow_experiment", "test_run_name", "test_dataset")

    N_CLASSES = 10
    N_SAMPLES = 1000
    BATCH_SIZE = 100
    EPOCHS = 10

    def gen_numpy_training_data(
        n_classes=10,
        n_samples=1000,
        noise_std=0.1,
        convert_to_bolt_dataset=True,
        batch_size_for_conversion=64,
    ):
        possible_one_hot_encodings = np.eye(n_classes)
        labels = np.random.choice(n_classes, size=n_samples).astype("uint32")
        examples = possible_one_hot_encodings[labels]
        noise = np.random.normal(0, noise_std, examples.shape)
        examples = (examples + noise).astype("float32")
        if convert_to_bolt_dataset:
            examples = dataset.from_numpy(
                examples, batch_size=batch_size_for_conversion
            )
            labels = dataset.from_numpy(labels, batch_size=batch_size_for_conversion)
        return examples, labels

    data, labels = gen_numpy_training_data(
        n_classes=N_CLASSES,
        n_samples=N_SAMPLES,
        noise_std=0.3,
        convert_to_bolt_dataset=True,
        batch_size_for_conversion=BATCH_SIZE,
    )

    def get_simple_dag_model(
        input_dim,
        hidden_layer_dim,
        hidden_layer_sparsity,
        output_dim,
        output_activation="softmax",
        loss=bolt.CategoricalCrossEntropyLoss(),
    ):
        input_layer = bolt.graph.Input(dim=input_dim)

        hidden_layer = bolt.graph.FullyConnected(
            dim=hidden_layer_dim, sparsity=hidden_layer_sparsity, activation="relu"
        )(input_layer)

        output_layer = bolt.graph.FullyConnected(
            dim=output_dim, activation=output_activation
        )(hidden_layer)

        model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

        model.compile(loss)

        return model

    model = get_simple_dag_model(
        input_dim=N_CLASSES,
        hidden_layer_dim=2000,
        hidden_layer_sparsity=1.0,
        output_dim=N_CLASSES,
    )

    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=EPOCHS)
        .with_metrics(["categorical_accuracy"])
        .with_callbacks([mlflowcallback])
    )

    return model.train(data, labels, train_config)


@pytest.mark.unit
def test_mach_save_load():
    num_train = 100
    num_test = 100
    num_true_labels_per_sample = 10
    input_and_output_dim = 100

    mach = build_and_train_mach(
        num_train=num_train,
        num_true_labels_per_sample=num_true_labels_per_sample,
        input_and_output_dim=input_and_output_dim,
        learning_rate=0.001,
        batch_size=512,
        num_epochs=5,
    )

    test_x_np, test_y_np = generate_random_easy_sparse(
        output_dim=input_and_output_dim,
        num_true_labels_per_example=num_true_labels_per_sample,
        num_examples=num_test,
    )

    result_fast, _ = mach.query_fast(test_x_np)
    result_slow, _ = mach.query_slow(test_x_np)
    recall_fast_before_save = get_recall(
        result_fast, test_y_np, num_true_labels_per_sample
    )
    recall_slow_before_save = get_recall(
        result_slow, test_y_np, num_true_labels_per_sample
    )

    save_folder_name = "mach_saved_for_test"
    mach.save(save_folder_name)

    newMach = bolt.Mach.load(save_folder_name)

    assert recall_fast_before_save == get_recall(
        newMach.query_fast(test_x_np)[0], test_y_np, num_true_labels_per_sample
    )
    assert recall_slow_before_save == get_recall(
        newMach.query_slow(test_x_np)[0], test_y_np, num_true_labels_per_sample
    )

    shutil.rmtree(save_folder_name)


@pytest.mark.unit
def test_mach_random_data():

    num_train = 10000
    num_test = 1000
    num_true_labels_per_sample = 10
    input_and_output_dim = 1000

    mach = build_and_train_mach(
        num_train=num_train,
        num_true_labels_per_sample=num_true_labels_per_sample,
        input_and_output_dim=input_and_output_dim,
        learning_rate=0.001,
        batch_size=512,
        num_epochs=5,
    )

    test_x_np, test_y_np = generate_random_easy_sparse(
        output_dim=input_and_output_dim,
        num_true_labels_per_example=num_true_labels_per_sample,
        num_examples=num_test,
    )

    result_fast, _ = mach.query_fast(test_x_np)
    result_slow, _ = mach.query_slow(test_x_np)

    assert get_recall(result_fast, test_y_np, num_true_labels_per_sample) > 0.8
    assert get_recall(result_slow, test_y_np, num_true_labels_per_sample) > 0.8
