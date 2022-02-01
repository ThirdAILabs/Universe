import numpy as np
import time
import thirdai
import argparse

parser = argparse.ArgumentParser(description="Run MagSearch VGG Image Net Benchmark.")
parser.add_argument(
    "data_folder", help="The path to the folder containing the 129 imagenet .npy files."
)
parser.add_argument(
    "output_folder",
    help="The path to the folder where this script saves graphs and text results.",
)
args = parser.parse_args()

top_k_gt = 10
top_k_search = 100
max_numpy_chunk_exclusive = 129


def run_trial(reservoir_size, hashes_per_table, num_tables):
    print(
        f"Creating index with reservoir size {res} and {tables} hash"
        + f" tables, with {per_table} concatenated hashes per table.",
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
        batch = np.load("%s/chunk-ave%d.npy" % (args.data_folder, chunk_num))
        num_bytes += batch.nbytes
        index.add(dense_data=batch, starting_index=num_vectors)
        num_vectors += len(batch)
    end = time.perf_counter()
    indexing_time = end - start
    print(
        "Loading and indexing %d vectors (%d bytes) took:%f"
        % (num_bytes, num_vectors, end - start),
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
        res,
        hashes_per_table,
        num_tables,
        indexing_time,
        querying_time,
        total_recall,
    )


def get_pareto(values):
    results = []
    for v in sorted(values):
        if len(results) == 0 or results[-1][1] < v[1]:
            results.append(v)
    return results


reservoir_sizes = [100, 200, 500]
hashes_per_table = [12, 14, 16]
num_tables = [10, 50, 100, 200, 500]
results = []
for res in reservoir_sizes:
    for per_table in hashes_per_table:
        for tables in num_tables:
            results.append(run_trial(res, per_table, tables))

query_time_v_recall = [(result[4], result[5]) for result in results]
pareto = get_pareto(query_time_v_recall)

fastest_with_recall_80_percent = [p for p in pareto if p[1] > 0.8][0]
with open(args.output_folder + "/imagenet_results.txt") as f:
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
