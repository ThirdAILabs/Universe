
from _deps.ColBERT.colbertmodeling.checkpoint import Checkpoint
import pathlib
import numpy as np

class DocSearchModel:
    def __init__(self, path):
        checkpoint_path = f"{path}/checkpoint"
        self.checkpoint = Checkpoint(checkpoint_path).cpu()
        self.centroids = np.load(f"{path}/centroids.npy")

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encodeQuery(self, query):
        return self.checkpoint.queryFromText([query])[0]

    def encodeDocs(self, docs):
        return self.checkpoint.docFromText(docs)[0]

    def getCentroids(self):
        return self.centroids
