from thirdai import search, utils
from timeit import default_timer as timer
import argparse

parser = argparse.ArgumentParser(description='Benchmark Flash on webspam.')
parser.add_argument('webspam_data_path',
                    help='the file name of the unzipped webspam svm data file')
parser.add_argument('webspam_query_path',
                    help='the file name of the unzipped webspam svm query file')
parser.add_argument('webspam_gt_path',
                    help='the file name of the unzipped webspam ground truth file')
args = parser.parse_args()


dataset = utils.loadInMemorySvmDataset(args.webspam_data_path, batch_size=10000)
queries = utils.loadInMemorySvmDataset(args.webspam_query_path, batch_size=100000)

gt = []
gt_distances = []
with open(args.webspam_gt_path) as f:
  gt = [[int(i) for i in line.split()] for line in f.readlines()]

r1_at_100_recall_time_pairs = []
max_size_bytes = 6000000000
table_range = 1000000
num_hashes_per_table = 4
max_num_threads_for_indexing = 50
utils.set_global_num_threads(max_num_threads_for_indexing)

for tables in [2 ** exp for exp in range(1, 9)]:
  for res_size in [2 ** exp for exp in range(1, 8)]:       
    hf = utils.MinHash(num_tables=tables, hashes_per_table=num_hashes_per_table, range=table_range)
    flash = search.Flash(hf, reservoir_size=res_size)
    flash.add_dataset(dataset)

    # Make these tests single threaded so we can better compare performance 
    utils.set_global_num_threads(1)
    start = timer()
    results = flash.query_batch(queries[0], top_k=100)
    end = timer()
    utils.set_global_num_threads(max_num_threads_for_indexing)

    r1_at_100 = 0
    for i, r in enumerate(results):
      r1_at_100 += gt[i][0] in r
    r1_at_100 /= len(results)

    r1_at_100_recall_time_pairs.append((r1_at_100, (end - start) / len(results)))
    print(r1_at_100_recall_time_pairs[-1], res_size, tables, flush=True)

def get_pareto_front(pairs):
	pairs = sorted(pairs, key=lambda pair: pair[1])
	result = []
	for pair in pairs:
		if len(result) == 0 or pair[0] > result[-1][0]:
			result.append(pair)
	return result

pareto = get_pareto_front(r1_at_100_recall_time_pairs)
print(pareto)

import matplotlib.pyplot as plt
import math

plt.plot([p[0] for p in pareto], [math.log10(1 / p[1]) for p in pareto])

titlefontsize = 22
axisfontsize = 18
labelfontsize = 12
ls = "--"

plt.xlabel("Recall (R1@100)", fontsize=axisfontsize)
plt.ylabel('Queries per second (log 10)', fontsize=axisfontsize)
plt.title("Flash Recall on Webspam", fontsize=titlefontsize)
plt.show()

# TODO(josh) report: R1@1, R1@10, R1@100, R10@10, R10@100, R100@100 so
# we can make this a benchmarking test
