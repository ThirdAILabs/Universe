import numpy as np
import time
import thirdai
import argparse
import mlflow

# Add the logging folder to the system path
import sys

sys.path.insert(1, sys.path[0] + "/../../logging/")
from mlflow_logger import log_imagesearch_run

parser = argparse.ArgumentParser(description="Run MagSearch VGG Image Net Benchmark.")
parser.add_argument(
    "--data_folder",
    help="The folder containing the 129 imagenet .npy files.",
    required=False,
    default="/media/scratch/data/ImageNet",
)
parser.add_argument(
    "--read_in_entire_dataset",
    action="store_true",
    help="Speed up program is we have enough memory to read in the entire dataset (~40 GB)",
)
args = parser.parse_args()

top_k_gt = 10
top_k_search = 100
max_numpy_chunk_exclusive = 129


def load_ith_numpy_batch(i):
    return np.load("%s/chunk-ave%d.npy" % (args.data_folder, i))


if args.read_in_entire_dataset:
    all_batches = [
        load_ith_numpy_batch(chunk_num)
        for chunk_num in range(max_numpy_chunk_exclusive)
    ]


def get_ith_batch(i):
    if args.read_in_entire_dataset:
        return all_batches[i]
    else:
        return load_ith_numpy_batch(i)


def run_trial(reservoir_size, hashes_per_table, num_tables):

    hf = thirdai.hashing.SignedRandomProjection(
        input_dim=4096, hashes_per_table=hashes_per_table, num_tables=num_tables
    )
    index = thirdai.search.MagSearch(hf, reservoir_size=reservoir_size)

    start = time.perf_counter()
    num_vectors = 0
    for chunk_num in range(0, max_numpy_chunk_exclusive):
        batch = get_ith_batch(chunk_num)
        index.add(dense_data=batch, starting_index=num_vectors)
        num_vectors += len(batch)
    end = time.perf_counter()
    indexing_time = end - start
    print(indexing_time, flush=True)

    queries = np.load(args.data_folder + "/test_embeddings.npy")
    start = time.perf_counter()
    results = index.query(queries, top_k=top_k_search)
    end = time.perf_counter()
    querying_time = end - start
    print(querying_time, flush=True)

    gt = np.load(args.data_folder + "/ground_truth.npy")
    recals = [
        sum(gt[i] in result for i in range(top_k_gt)) / top_k_gt
        for result, gt in zip(results, gt)
    ]
    total_recall = sum(recals) / len(recals)

    return indexing_time, querying_time, total_recall


# From a larger grid search, these were a good representative of the best
# hyperparameters. For intuition, for low recall to optimize speed we choose
# a lot of hash functions and increasing number of hash tables, along with a
# reasonably small reservoir. Towards the higher end of recall, we decrease
# the number of hashes per table to try to find more neighbors, and increase
# reservoir size.
trials = [
    (10, 16, 100),
    (25, 16, 100),
    (50, 16, 100),
    (100, 16, 100),
    (200, 16, 100),
    (500, 16, 200),
    (500, 14, 200),
    (500, 12, 500),
    (1000, 10, 1000),
]

for (num_tables, hashes_per_table, reservoir_size) in trials:
    indexing_time, querying_time, recall = run_trial(
        reservoir_size, hashes_per_table, num_tables
    )
    log_imagesearch_run(
        reservoir_size=reservoir_size,
        hashes_per_table=hashes_per_table,
        num_tables=num_tables,
        indexing_time=indexing_time,
        querying_time=querying_time,
        num_queries=10000,
        recall=recall,
        dataset="imagenet_embeddings",
    )
