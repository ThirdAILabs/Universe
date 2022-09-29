try:
    import numpy as np
    import torch
    import transformers
except ImportError as e:
    print(
        "The embeddings package requires the PyTorch, transformers, and numpy "
        "packages. Please install these before importing the embeddings "
        "package by e.g. running `pip3 install torch transformers numpy`."
    )
    raise e

from thirdai._deps.ColBERT.colbertmodeling.checkpoint import Checkpoint
from thirdai._download import ensure_targz_installed

MSMARCO_URL = "https://www.dropbox.com/s/s02nev64icelbkr/msmarco.tar.gz?dl=0"
MSMARCO_DIR_NAME = "msmarco"


class DocSearchModel:
    def __init__(
        self, local_path=None, download_metadata=(MSMARCO_URL, MSMARCO_DIR_NAME)
    ):
        if not local_path:
            local_path, _ = ensure_targz_installed(
                download_url=download_metadata[0],
                unzipped_dir_name=download_metadata[1],
            )
        self.checkpoint = Checkpoint(str(local_path)).cpu()
        self.centroids = np.load(f"{local_path}/centroids.npy")

    def encodeQuery(self, query):
        return self.checkpoint.queryFromText([query])[0]

    def encodeDocs(self, docs):
        return self.checkpoint.docFromText(docs)

    def getCentroids(self):
        return self.centroids
