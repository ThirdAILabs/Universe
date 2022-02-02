import numpy as np
import time
import thirdai
import argparse

parser = argparse.ArgumentParser(description="Run MagSearch VGG Image Net Benchmark.")
parser.add_argument(
    "output_folder",
    help="The folder where this script should save graphs and text results.",
)
parser.add_argument(
    "data_folder", help="The folder containing the 129 imagenet .npy files."
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
    print(
        f"Creating index with reservoir size {reservoir_size} and {num_tables} hash"
        + f" tables, with {hashes_per_table} concatenated hashes per table.",
        flush=True,
    )

    hf = thirdai.hashing.SignedRandomProjection(
        input_dim=4096, hashes_per_table=hashes_per_table, num_tables=num_tables
    )
    index = thirdai.search.MagSearch(hf, reservoir_size=reservoir_size)

    num_vectors = 0
    start = time.perf_counter()
    num_bytes = 0
    for chunk_num in range(0, max_numpy_chunk_exclusive):
        batch = get_ith_batch(chunk_num)
        num_bytes += batch.nbytes
        index.add(dense_data=batch, starting_index=num_vectors)
        num_vectors += len(batch)
    end = time.perf_counter()
    indexing_time = end - start
    print(
        "Loading and indexing %d vectors (%d bytes) took:%f"
        % (num_vectors, num_bytes, end - start),
        flush=True,
    )

    queries = np.load(args.data_folder + "/test_embeddings.npy")
    start = time.perf_counter()
    results = index.query(queries, top_k=top_k_search)
    end = time.perf_counter()
    querying_time = end - start
    print(
        "Querying %d vectors took:%f" % (len(queries), querying_time),
        flush=True,
    )

    gt = np.load(args.data_folder + "/ground_truth.npy")
    recals = [
        sum(gt[i] in result for i in range(top_k_gt)) / top_k_gt
        for result, gt in zip(results, gt)
    ]
    total_recall = sum(recals) / len(recals)
    print(f"R{top_k_gt}@{top_k_search} = {total_recall}", flush=True)
    return (
        reservoir_size,
        hashes_per_table,
        num_tables,
        indexing_time,
        querying_time,
        total_recall,
    )


reservoir_sizes = [100, 200, 500]
hashes_per_table = [12, 14, 16]
num_tables = [10, 50, 100, 200, 500]
results = []

# From a larger grid search, these were a good representative of the best
# hyperparameters. For intuition, for low recall to optimize speed we choose
# a lot of hash functions and increasing number of hash tables, along with a 
# reasonably small reservoir. Towards the higher end of recall, we decrease
# the number of hashes per table to try to find more neighbors, and increase
# reservoir size.
trials = [
  (10, 16, 100), (25, 16, 100), (50, 16, 100), (100, 16, 100), (200, 16, 100), 
  (500, 16, 200), (500, 14, 200), (500, 12, 500), (1000, 10, 1000)]

for (num_tables, hashes_per_table, reservoir_size) in trials:
    results.append(run_trial(reservoir_size, hashes_per_table, num_tables))


def get_pareto(values):
    results = []
    for v in sorted(values):
        if len(results) == 0 or results[-1][1] < v[1]:
            results.append(v)
    return results


query_time_v_recall = [(result[4], result[5]) for result in results]
pareto = get_pareto(query_time_v_recall)

fastest_with_recall_80_percent = [p for p in pareto if p[1] > 0.8][0]
with open(args.output_folder + "/imagenet_results.txt", "w") as f:
    f.write(
        f"""
	Succesfully ran {len(results)} trials of MagSearch on the ImageNet embedding
	dataset. The fastest trial with a recall of 0.8 took 
	{fastest_with_recall_80_percent} seconds for 10,000 queries 
	({10000 / fastest_with_recall_80_percent} queries per second). """
    )


import matplotlib.pyplot as plt
import math

plt.plot(
    [p[1] for p in pareto],
    [math.log10(10000 / p[0]) for p in pareto],
)
titlefontsize = 22
axisfontsize = 18
labelfontsize = 12
ls = "--"
plt.xlabel("Recall (R10@100)", fontsize=axisfontsize)
plt.ylabel("Queries per second (log 10)", fontsize=axisfontsize)
plt.title("Flash Recall on ImageNet Vgg Embeddings", fontsize=titlefontsize)
plt.savefig(args.output_folder + "/r10_at_100_recall_vs_time.png")
