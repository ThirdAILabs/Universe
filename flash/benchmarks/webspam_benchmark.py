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

queries = utils.load_svm(args.webspam_query_path, batch_size=100000, batches_per_load=1)
queries.LoadNextSetOfBatches()

gt = []
with open("../webspam_gt") as f:
  gt = [[int(i) for i in line.split()] for line in f.readlines()]


# Make these tests single threaded so we can better compare performance 
utils.set_global_num_threads(1)


r1_at_100_recall_time_pairs = []
max_size_bytes = 10000000000
table_range = 1000000
for j in range(1, 10):
    tables = 2**j 
    k = 4

    # TODO: We have to make reservoirs smaller as we go up to not go OOM
    # on a laptop, see https://github.com/ThirdAILabs/Universe/issues/133 for a 
    # planned fix
    reservoir_size = min(200, max_size_bytes // table_range // tables)


    # TODO(josh): Read this in only once when we refactor the dataset class
    hf = utils.DensifiedMinHash(num_tables=tables, hashes_per_table=k, range=table_range)
    dataset = utils.load_svm(args.webspam_data_path, batch_size=10000, batches_per_load=1)

    flash = search.Flash(hf, reservoir_size)
    flash.AddDataset(dataset)

    start = timer()
    results = flash.QueryBatch(queries[0], top_k=100)
    end = timer()

    r1_at_100 = 0
    for i, r in enumerate(results):
      r1_at_100 += gt[i][0] in r
    r1_at_100 /= len(results)

    r1_at_100_recall_time_pairs.append((r1_at_100, (end - start) / len(results)))
    print(r1_at_100_recall_time_pairs, flush=True)


