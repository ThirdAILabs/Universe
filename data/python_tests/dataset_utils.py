import random
import string

import numpy as np
from thirdai import data


def random_word(length=4):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def random_sentence(num_words):
    return " ".join(random_word() for _ in range(num_words))


def get_random_str_column(col_length):
    return data.columns.StringColumn([random_word() for _ in range(col_length)])


def get_random_sentence_str_column(col_length, num_words):
    return data.columns.StringColumn(
        [random_sentence(num_words) for _ in range(col_length)]
    )


def verify_hash_distribution(all_hashes, output_range):
    hash_counts = [0 for _ in range(output_range)]
    for hashes in all_hashes:
        for h in hashes:
            hash_counts[h] += 1

    expected_count = sum(hash_counts) / output_range
    for count in hash_counts:
        assert count / expected_count < 2 and count / expected_count > 0.5


def get_ascending_column_map(rows, offset=0):
    end = rows + offset
    return data.ColumnMap(
        {
            "token": data.columns.TokenColumn(list(range(offset, end))),
            "decimal": data.columns.DecimalColumn(list(range(offset, end))),
            "token_array": data.columns.TokenArrayColumn(
                [list(range(i, i + 4)) for i in range(offset, end)]
            ),
            "decimal_array": data.columns.DecimalArrayColumn(
                [list(range(i, i + 4)) for i in range(offset, end)]
            ),
            "string": data.columns.StringColumn([str(i) for i in range(offset, end)]),
        }
    )
