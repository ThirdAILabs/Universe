import random
import string

import numpy as np
from thirdai import data


def dense_vectors_to_numpy(vectors):
    return np.array([v.to_numpy() for v in vectors])


def sparse_vectors_to_numpy(vectors):
    indices_list = []
    values_list = []
    for vec in vectors:
        (i, v) = vec.to_numpy()
        indices_list.append(i)
        values_list.append(v)

    indices = np.array(indices_list)
    values = np.array(values_list)
    return (indices, values)


def get_bolt_vectors_from_dataset(dataset):
    vectors = []
    for batch in range(len(dataset)):
        for vec in range(len(dataset[batch])):
            vectors.append(dataset[batch][vec])
    return vectors


def dense_bolt_dataset_to_numpy(dataset):
    return dense_vectors_to_numpy(get_bolt_vectors_from_dataset(dataset))


def sparse_bolt_dataset_to_numpy(dataset):
    return sparse_vectors_to_numpy(get_bolt_vectors_from_dataset(dataset))


def random_word(length=4):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def random_sentence(num_words):
    return " ".join(random_word() for _ in range(num_words))


def get_str_col(col_length):
    return data.columns.StringColumn([random_word() for _ in range(col_length)])


def get_sentence_str_column(col_length, num_words):
    return data.columns.StringColumn(
        [random_sentence(num_words) for _ in range(col_length)]
    )


# Given a sparse numpy dataset of featurized pairgrams (not deduplicated), count
# to make sure the number of pairgrams for each index across the whole dataset
# is close to the expected number based on the number of unigrams.
# If there are N unigrams, pairgrams should have (N * (N + 1)) / 2 values.
def verify_pairgrams(pairgram_dataset, output_range, num_unigrams):
    indices, values = pairgram_dataset
    hash_counts = [0 for _ in range(output_range)]
    for row_indices, row_values in zip(indices, values):
        for index, value in zip(row_indices, row_values):
            hash_counts[index] += value

    pairgrams_per_row = (num_unigrams * (num_unigrams + 1)) / 2
    expected_count = (len(indices) / output_range) * pairgrams_per_row
    for count in hash_counts:
        assert count / expected_count < 2 and count / expected_count > 0.5


# Given a sparse numpy dataset of featurized unigrams (not deduplicated), count
# to make sure the number of unigrams for each index across the whole dataset
# is close to the expected number.
def verify_unigrams(pairgram_dataset, output_range, expected_unigrams_per_row):
    indices, values = pairgram_dataset
    hash_counts = [0 for _ in range(output_range)]
    for row_indices, row_values in zip(indices, values):
        for index, value in zip(row_indices, row_values):
            hash_counts[index] += value

    expected_num_unigrams = expected_unigrams_per_row * len(indices) / output_range
    for count in hash_counts:
        assert count / expected_num_unigrams < 2 and count / expected_num_unigrams > 0.5
