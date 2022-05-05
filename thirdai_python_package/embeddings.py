try:
    import torch
    import transformers
    import numpy as np
except ImportError as e:
    print(
        "The embeddings package requires the PyTorch, transformers, and numpy "
        "packages. Please install these before importing the embeddings "
        "package by e.g. running `pip3 install torch transformers`."
    )
    raise e

from ._deps.ColBERT.colbertmodeling.checkpoint import Checkpoint
import pathlib
import os

CACHE_DIR = pathlib.Path.home() / ".cache" / "thirdai"
MSMARCO_MODEL_URL = "https://www.dropbox.com/s/s02nev64icelbkr/msmarco.tar.gz?dl=0"
MSMARCO_DIR = CACHE_DIR / "msmarco"
MSMARCO_DOWNLOAD_PATH = CACHE_DIR / "msmarco.tar.gz"


def ensure_msmarco_model_installed():
    # TODO(josh): This isn't really robust (it relies on the user having curl
    # and tar working on their machine), but the python requests library wasn't
    # working to download from dropbox so I am just going with this for now.
    # We should make this a more robust solution like huggingface's dataset
    # library.
    if not MSMARCO_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.system(f"curl -L {MSMARCO_MODEL_URL} -o {MSMARCO_DOWNLOAD_PATH}")
        os.system(f"tar -xzf {MSMARCO_DOWNLOAD_PATH} -C {CACHE_DIR}")
        MSMARCO_DOWNLOAD_PATH.unlink()


class DocSearchModel:
    def __init__(self, path=None):
        if not path:
            ensure_msmarco_model_installed()
            path = MSMARCO_DIR
        self.checkpoint = Checkpoint(str(path)).cpu()
        self.centroids = np.load(f"{path}/centroids.npy")

    def encodeQuery(self, query):
        return self.checkpoint.queryFromText([query])[0]

    def encodeDocs(self, docs):
        return self.checkpoint.docFromText(docs)[0]

    def getCentroids(self):
        return self.centroids
