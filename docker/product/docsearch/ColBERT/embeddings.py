from modeling.checkpoint import Checkpoint
import pathlib
import numpy as np


class DocSearchModel:
    def __init__(self):
        path_to_this_file = pathlib.Path(__file__).parent.resolve()
        checkpoint_path = str(path_to_this_file) + "/downloads/colbertv2.0"
        self.checkpoint = Checkpoint(checkpoint_path).cpu()
        self.centroids = np.load(checkpoint_path + "/downloads/centroids.npy")

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encodeQuery(self, query):
        return self.checkpoint.queryFromText([query]).numpy()[0]

    def encodeDocs(self, docs):
        return self.checkpoint.docFromText(docs).numpy()

    def getCentroids(self):
        return self.centroids
