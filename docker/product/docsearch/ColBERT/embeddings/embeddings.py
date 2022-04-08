from colbertmodeling.checkpoint import Checkpoint
import pathlib
import numpy as np


class DocSearchModel:
    def __init__(self):
        checkpoint_path = "/home/thirdai/saved/downloads/colbertv2.0"
        self.checkpoint = Checkpoint(checkpoint_path).cpu()
        self.centroids = np.load("/home/thirdai/saved/downloads/centroids.npy")

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encodeQuery(self, query):
        return self.checkpoint.queryFromText([query]).numpy()[0]

    def encodeDocs(self, docs):
        return self.checkpoint.docFromText(docs).numpy()

    def getCentroids(self):
        return self.centroids
