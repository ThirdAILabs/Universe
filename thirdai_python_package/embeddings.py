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
import os

cache_dir = pathlib.Path.home() / ".cache" / "thirdai"
msmarco_model_url = "https://www.dropbox.com/s/s02nev64icelbkr/msmarco.tar.gz?dl=0"
msmarco_dir = cache_dir / "msmarco"
msmarco_download_path = cache_dir / "msmarco.tar.gz"


def ensure_msmarco_model_installed():
    # TODO(josh): This isn't really robust (it relies on the user having curl
    # and tar working on their machine), but the python requests library wasn't
    # working to download from dropbox so I am just going with this for now.
    # We should make this a more robust solution like huggingface's dataset
    # library.
    if not msmarco_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.system(f"curl -L {msmarco_model_url} -o {msmarco_download_path}")
        os.system(f"tar -xzf {msmarco_download_path} -C {cache_dir}")
        msmarco_download_path.unlink()


class DocSearchModel:
    def __init__(self, path=None):
        if not path:
            ensure_msmarco_model_installed()
            path = msmarco_dir
        self.checkpoint = Checkpoint(str(path)).cpu()
        self.centroids = np.load(f"{path}/centroids.npy")

    def encodeQuery(self, query):
        return self.checkpoint.queryFromText([query])[0]

    def encodeDocs(self, docs):
        return self.checkpoint.docFromText(docs)[0]

    def getCentroids(self):
        return self.centroids
