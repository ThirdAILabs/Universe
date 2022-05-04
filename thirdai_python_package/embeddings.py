try:
    import torch
    import transformers
except ImportError as e:
    print(
        "The embeddings package requires the PyTorch and Transformers "
        "packages. Please install these before importing the embeddings "
        "package by e.g. running `pip3 install torch transformers`."
    )
    raise e

from ._deps.ColBERT.colbertmodeling.checkpoint import Checkpoint
import pathlib
import numpy as np


class DocSearchModel:
    def __init__(self, path):
        checkpoint_path = f"{path}/checkpoint"
        self.checkpoint = Checkpoint(checkpoint_path).cpu()
        self.centroids = np.load(f"{path}/centroids.npy")

    def encodeQuery(self, query):
        return self.checkpoint.queryFromText([query])[0]

    def encodeDocs(self, docs):
        return self.checkpoint.docFromText(docs)[0]

    def getCentroids(self):
        return self.centroids
