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


dataset = utils.InMemorySparseDataset(args.webspam_data_path, batch_size=10000)
queries = utils.InMemorySparseDataset(args.webspam_query_path, batch_size=100000)

gt = []
with open("../webspam_gt") as f:
  gt = [[int(i) for i in line.split()] for line in f.readlines()]

r1_at_100_recall_time_pairs = []
max_size_bytes = 6000000000
table_range = 1000000
for j in range(1, 10):
    for res_size in [1000, 2000, 4000, 8000]:
      for k in [4, 8, 12]:
        tables = 2**j 
        # k = 4
        # hf = utils.MinHash(num_tables=tables, hashes_per_table=k, range=table_range)
        hf = utils.SignedRandomProjection(num_tables=tables, hashes_per_table=k, input_dim=16609143)
        flash = search.Flash(hf, reservoir_size=res_size)
        flash.add_dataset(dataset)

        # Make these tests single threaded so we can better compare performance 
        utils.set_global_num_threads(1)
        start = timer()
        results = flash.query_batch(queries[0], top_k=100)
        end = timer()
        utils.set_global_num_threads(50)

        r1_at_100 = 0
        for i, r in enumerate(results):
          r1_at_100 += gt[i][0] in r
        r1_at_100 /= len(results)

        r1_at_100_recall_time_pairs.append((r1_at_100, (end - start) / len(results)))
        print(r1_at_100_recall_time_pairs[-1], res_size, k, flush=True)