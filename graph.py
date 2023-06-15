import sys

import h5py
import numpy as np
from thirdai import search

filename = sys.argv[1]

f = h5py.File(filename, "r")

train = f.get("train")[()]
test = f.get("test")[()]
gtruth = f.get("neighbors")[()]

print(train.shape, train.dtype)
print(test.shape, test.dtype)
print(gtruth.shape, gtruth.dtype)


index = search.HNSW(max_nbrs=16, data=train, construction_buffer_size=32)


k = 100
recall = 0.0
for row, actual in zip(test, gtruth):
    nbrs = np.array(index.query(query=row, k=k, search_buffer_size=128))
    recall += len(np.intersect1d(nbrs, actual[:k])) / k


print("Mean recall: ", recall / len(test))
