from thirdai.dataset import TextCustomDense, Text
from core import Dataset, Schema
from sources import InMemoryCollection
from parsers import CsvIterable
import numpy as np
import time

class NumpyMaker:
    def __init__(self):
        self.counter = 0
    def make_numpy(self, text):
        nparr = np.array([self.counter for _ in range(10)])
        self.counter += 1
        return nparr

# TODO
# Geordie, fix segfault.
# Functions fine when 8000, fails when 10000
# Should work fine when 8192 and fail when 8193 because size of processing batch is 8192.
# Well I'll be damned. It breaks at 8192.
# And 8000!
# And 1000!
# And 500!!!
# And 256...128...
# Probably something to do with something not being kept alive.

"""
terminate called after throwing an instance of 'pybind11::error_already_set'
  what():  RuntimeError: Inconsistent object during array creation? Content of sequences changed (length inconsistent).

At:
  /Users/benitogeordie/Desktop/Universe/dataset/test2.py(12): make_numpy
  /Users/benitogeordie/Desktop/Universe/data/src/core/dataset.py(143): __load_all_and_process
  /Users/benitogeordie/Desktop/Universe/data/src/core/dataset.py(185): processInMemory
  /Users/benitogeordie/Desktop/Universe/dataset/test2.py(41): <module>
"""

fake_list = [
    "one" for i in range(128)
]

npm = NumpyMaker()

dataset = Dataset(
    source=InMemoryCollection(fake_list),
    parser=CsvIterable(),
    schema=Schema(input_blocks=[Text(col=0, embedding=TextCustomDense(dim=10, embed_fn=npm.make_numpy))])
)

start = time.time()
print("Starting")
print(dataset.processInMemory().at(0).to_string())
end = time.time()
elapsed = end - start
print(f"Done in {elapsed} seconds.")


