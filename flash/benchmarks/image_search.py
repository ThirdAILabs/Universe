import numpy as np
import time
import thirdai
import sklearn
from sklearn import metrics

data_path = "/Users/josh/IndexChunks/"
top_k_gt = 10
top_k_flash = 100

hf = thirdai.utils.SignedRandomProjection(input_dim=4096, hashes_per_table=13, num_tables=500)
flash = thirdai.search.Flash(hf, reservoir_size=100)

max_chunk = 129
num_vectors = 0
start = time.perf_counter()
for chunk_num in range(0, max_chunk):
    batch = np.load("%schunk-ave%d.npy" % (data_path, chunk_num))
    flash.add(dense_data=batch, starting_index=num_vectors)
    num_vectors += len(batch)
end = time.perf_counter()
print("Loading and indexing %d vectors (40GB) took %f seconds." % (num_vectors, end - start))

queries = np.load(data_path + "test_embeddings.npy")
start = time.perf_counter()
results = flash.query(queries, top_k=top_k_flash)
end = time.perf_counter()
print("Querying %d vectors took %f seconds (%fms per query)." % (len(queries), end - start, (end - start) / len(queries) * 1000))

gt = np.load(data_path + "ground_truth.npy")
recals = [sum(gt[i] in result for i in range(top_k_gt)) / top_k_gt for result, gt in zip(results, gt)]
print(f"R{top_k_gt}@{top_k_flash} = {sum(recals) / len(recals)}")