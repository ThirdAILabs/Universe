import numpy as np
import time
import thirdai
import sklearn
from sklearn import metrics

reservoir_sizes = [100, 200, 500, 1000]
hashes_per_table = [8, 10, 12, 14, 16]
num_tables = [10, 50, 100, 200, 500]

# data_path = "/Users/josh/IndexChunks/"
data_path = "/media/scratch/ImageNetDemo/IndexFiles/"

for res in reservoir_sizes:
  for per_table in hashes_per_table:
    for tables in num_tables:

      print((res, per_table, tables), flush=True)

      top_k_gt = 10
      top_k_flash = 100

      hf = thirdai.hashing.SignedRandomProjection(input_dim=4096, hashes_per_table=per_table, num_tables=tables)
      flash = thirdai.search.MagSearch(hf, reservoir_size=res)

      max_chunk = 129
      num_vectors = 0
      start = time.perf_counter()
      for chunk_num in range(0, max_chunk):
          batch = np.load("%schunk-ave%d.npy" % (data_path, chunk_num))
          flash.add(dense_data=batch, starting_index=num_vectors)
          num_vectors += len(batch)
      end = time.perf_counter()
      print("Loading and indexing %d vectors (40GB) took:%f" % (num_vectors, end - start), flush=True)

      queries = np.load(data_path + "test_embeddings.npy")
      start = time.perf_counter()
      results = flash.query(queries, top_k=top_k_flash)
      end = time.perf_counter()
      print("Querying %d vectors took:%f" % (len(queries), end - start), flush=True)

      gt = np.load(data_path + "ground_truth.npy")
      recals = [sum(gt[i] in result for i in range(top_k_gt)) / top_k_gt for result, gt in zip(results, gt)]
      print(f"R{top_k_gt}@{top_k_flash} = {sum(recals) / len(recals)}", flush=True)

