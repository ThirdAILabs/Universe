from hashlib import new
import numpy as np
from thirdai import bolt
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

    train_x, train_y = generate_random_easy_sparse(
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
        train_x,
        train_y,
        learning_rate=learning_rate,
        batch_size=batch_size,
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
def test_mach_save_load():
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

    test_x, test_y = generate_random_easy_sparse(
        output_dim=input_and_output_dim,
        num_true_labels_per_example=num_true_labels_per_sample,
        num_examples=num_test,
    )

    result_fast = mach.query_fast(test_x)
    result_slow = mach.query_slow(test_x)
    recall_fast_before_save = get_recall(
        result_fast, test_y, num_true_labels_per_sample
    )
    recall_slow_before_save = get_recall(
        result_slow, test_y, num_true_labels_per_sample
    )

    save_folder_name = "mach_saved_for_test"
    mach.save(save_folder_name)

    newMach = bolt.Mach.load(save_folder_name)

    assert recall_fast_before_save == get_recall(
        newMach.query_fast(test_x), test_y, num_true_labels_per_sample
    )
    assert recall_slow_before_save == get_recall(
        newMach.query_slow(test_x), test_y, num_true_labels_per_sample
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

    test_x, test_y = generate_random_easy_sparse(
        output_dim=input_and_output_dim,
        num_true_labels_per_example=num_true_labels_per_sample,
        num_examples=num_test,
    )

    result_fast = mach.query_fast(test_x)
    result_slow = mach.query_slow(test_x)

    assert get_recall(result_fast, test_y, num_true_labels_per_sample) > 0.8
    assert get_recall(result_slow, test_y, num_true_labels_per_sample) > 0.8
