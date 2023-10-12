from typing import List
import random
import time
import argparse
from thirdai import neural_db as ndb
from thirdai.neural_db.constraint_matcher import (
    ConstraintMatcher,
    ConstraintValue,
    to_filters,
)
import mlflow
import json


def strings_of_length(length, num_strings):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if num_strings > len(alphabet) ** length:
        raise ValueError(
            f"There are only {len(alphabet) ** length} strings of length {length}"
        )

    strings = ["" for _ in range(num_strings)]
    for i in range(num_strings):
        to_be_modded = i
        for _ in range(length):
            strings[i] += alphabet[to_be_modded % len(alphabet)]
            to_be_modded //= len(alphabet)
    return strings


def generate_item_metadata(
    metadata_fields: List[str], metadata_options: List[str], num_items: int
):
    metadata = [{field: ""} for field in metadata_fields] * num_items
    for i in range(num_docs):
        to_be_modded = i
        for field in metadata_fields:
            metadata[i][field] = metadata_options[to_be_modded % len(metadata_options)]
            to_be_modded //= len(metadata_options)
    return metadata


def random_filters(
    num_exact_filters: int,
    num_range_filters: int,
    range_size: int,
    metadata_fields: List[str],
    sorted_metadata_options: List[str],
):
    exact_filter_fields = random.choices(metadata_fields, k=num_exact_filters)
    range_filter_fields = list(set(metadata_fields).difference(exact_filter_fields))
    range_filter_fields = random.choices(range_filter_fields, k=num_range_filters)

    exact_filters = {
        field: random.choice(sorted_metadata_options) for field in exact_filter_fields
    }

    max_idx = len(sorted_metadata_options) - range_size
    random_indices = [random.randint(0, max_idx) for _ in range_filter_fields]
    range_filters = {
        field: ndb.InRange(
            sorted_metadata_options[idx], sorted_metadata_options[idx + range_size - 1]
        )
        for field, idx in zip(range_filter_fields, random_indices)
    }

    return {**exact_filters, **range_filters}


def generate_constraints(
    num_exact_filters: int,
    num_range_filters: int,
    range_size: int,
    metadata_fields: List[str],
    metadata_options: List[str],
    num_queries: int,
):
    if num_exact_filters + num_range_filters > len(metadata_fields):
        raise ValueError(
            "num_exact_filters + num_range_filters has to be less than or equal to the number of metadata fields."
        )
    if range_size > len(metadata_options):
        raise ValueError(
            "Number of metadata options must be greater than or equal to range_size."
        )

    sorted_options = sorted(metadata_options)
    metadata = [
        random_filters(
            num_exact_filters,
            num_range_filters,
            range_size,
            metadata_fields,
            sorted_options,
        )
        for _ in range(num_queries)
    ]

    return metadata


def benchmark(
    num_metadata_fields: int,
    num_options_per_field: int,
    metadata_field_len: int,
    metadata_option_len: int,
    num_exact_filters: int,
    num_range_filters: int,
    range_size: int,
    num_docs: int,
    num_queries: int,
):
    if num_docs < num_metadata_fields * num_options_per_field:
        raise ValueError(
            "Num docs must be at least num_metadata_fields x num_options_per field."
        )

    metadata_fields = strings_of_length(metadata_field_len, num_metadata_fields)
    metadata_options = strings_of_length(metadata_option_len, num_options_per_field)
    item_metadata = generate_item_metadata(metadata_fields, metadata_options, num_docs)
    dummy_item = 0

    constraint_matcher = ConstraintMatcher()

    start = time.time()
    for meta in item_metadata:
        constraint_matcher.index(
            item=dummy_item,
            constraints={key: ConstraintValue(value) for key, value in meta},
        )
    constraint_indexing_time = time.time() - start

    constraint_queries = generate_constraints(
        num_exact_filters,
        num_range_filters,
        range_size,
        metadata_fields,
        metadata_options,
        num_queries,
    )

    start = time.time()
    for constraints in constraint_queries:
        start = time.time()
        constraint_matcher.match(filters=to_filters(constraints))
    constraint_matching_time = time.time() - start

    return {
        "avg_constraint_matching_time": constraint_matching_time / num_queries,
        "total_constraint_indexing_time": constraint_indexing_time,
    }


def main():
    benchmark(
        num_metadata_fields=10,
        num_options_per_field=3,
        metadata_field_len=5,
        metadata_option_len=5,
        num_exact_filters=7,
        num_range_filters=3,
        range_size=3,
        num_docs=10_000_000,
        num_queries=1000,
    )
