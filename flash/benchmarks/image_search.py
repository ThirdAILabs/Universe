import numpy as np
import time
import thirdai

data_path = "/Users/josh/IndexChunks/"

hf = thirdai.utils.SignedRandomProjection(input_dim=4096, hashes_per_table=15, num_tables=300)
flash = thirdai.search.Flash(hf)

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
results = flash.query(queries)
end = time.perf_counter()
print("Querying %d vectors took %f seconds." % (len(queries), end - start))