from typing import List
import random
import string
from thirdai import neural_db as ndb


def strings_of_length(length, num_strings):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if num_strings > len(alphabet) ** length:
        raise ValueError(
            f"There are only {len(alphabet) ** length} strings of length {length}"
        )

    strings = ["a" * length] * num_strings
    for i in range(num_strings):
        to_be_modded = i
        for pos in range(length):
            strings[i][pos] = alphabet[to_be_modded % len(alphabet)]
            to_be_modded /= len(alphabet)
    return strings


def write_files(num_docs: int, num_entities_per_doc: int, entity: str):
    if "," in entity:
        raise ValueError("Entity cannot contain delimiter character ','")

    for i in range(num_docs):
        with open(f"doc_{i}.csv", "w") as f:
            f.write("id,text\n")
            for j in range(num_entities_per_doc):
                f.write(f"{j},{entity},\n")

    return [f"doc_{i}.csv" for i in range(num_docs)]


def assign_metadata_to_docs(
    metadata_fields: List[str], metadata_options: List[str], num_docs: int
):
    metadata = [{field: ""} for field in metadata_fields] * num_docs
    for i in range(num_docs):
        to_be_modded = i
        for field in metadata_fields:
            metadata[i][field] = metadata_options[to_be_modded % len(metadata_options)]
            to_be_modded /= len(metadata_options)
    return metadata


def random_filters(
    num_exact_filters: int,
    num_range_filters: int,
    range_size: int,
    metadata_fields: List[str],
    sorted_metadata_options: List[str],
):
    exact_filter_fields = random.choices(metadata_fields, k=num_exact_filters)
    range_filter_fields = set(metadata_fields).difference(exact_filter_fields)
    range_filter_fields = random.choices(range_filter_fields, k=num_range_filters)

    exact_filters = {
        field: random.choice(sorted_metadata_options) for field in exact_filter_fields
    }

    max_idx = len(sorted_metadata_options) - range_size + 1
    random_indices = [random.randint(0, max_idx) for _ in range_filter_fields]
    range_filters = {
        field: ndb.InRange(
            sorted_metadata_options[idx], sorted_metadata_options[idx + range_size - 1]
        )
        for field, idx in zip(range_filter_fields, random_indices)
    }

    return {**exact_filters, **range_filters}


def generate_queries(
    num_exact_filters: int,
    num_range_filters: int,
    range_size: int,
    metadata_fields: List[str],
    metadata_options: List[str],
    num_queries: int,
    query_length: int,
):
    if num_exact_filters + num_range_filters > len(metadata_fields):
        raise ValueError(
            "num_exact_filters + num_range_filters has to be less than or equal to the number of metadata fields."
        )
    if range_size < metadata_options:
        raise ValueError(
            "Number of metadata options must be greater than or equal to range_size."
        )

    queries = [
        "".join(random.choices(string.ascii_uppercase + string.digits, k=query_length))
        for _ in range(num_queries)
    ]

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

    return zip(queries, metadata)


def benchmark(
    num_metadata_fields: int,
    num_options_per_field: int,
    metadata_field_len: int,
    metadata_option_len: int,
    num_exact_filters: int,
    num_range_filters: int,
    range_size: int,
    num_docs: int,
    num_entities_per_doc: int,
    num_queries: int,
    entity: str,
):
    if num_docs < 10 * num_metadata_fields * num_options_per_field:
        raise ValueError(
            "Num docs must be at least 10 x num_metadata_fields x num_options_per field."
        )
    # Steps:
    # Generate metadata options
    metadata_fields = strings_of_length(metadata_field_len, num_metadata_fields)
    metadata_options = strings_of_length(metadata_option_len, num_options_per_field)
    filenames = write_files(num_docs, num_entities_per_doc, entity)
    doc_metadata = assign_metadata_to_docs(metadata_fields, metadata_options, num_docs)
    documents = [
        ndb.CSV(filename, metadata=meta)
        for filename, meta in zip(filenames, doc_metadata)
    ]

    db = ndb.NeuralDB()
    db.insert(documents, train=False)

    constraint_matching_time = 0
    scoring_time = 0

    for query, constraints in generate_queries(
        num_exact_filters, num_range_filters, num_queries, metadata_options
    ):
        _, times = db.search(query, constraints=constraints, return_times=True)
        constraint_matching_time += times["constraint_matching"]
        scoring_time += times["scoring"]

    avg_constraint_matching_time = constraint_matching_time / num_queries
    avg_scoring_time = scoring_time / num_queries

    print("avg_constraint_matching_time", avg_constraint_matching_time)
    print("avg_scoring_time", avg_scoring_time)
