from typing import List
import mmh3
import numpy as np
from .text_embedding_model_interface import TextEmbeddingModel
from utils.builder_vectors import __BuilderVector__


class TextOneHotEncoding(TextEmbeddingModel):
    """A text embedding model that uses hashes string tokens into integers,
    then one-hot encodes these integer tokens.
    """

    def __init__(self, output_dim: int, seed: int = 10):
        """Constructor.

        Arguments:
          output_dim: int - the output dimension of the embedding.
          seed: int - the seed used in the hash function.
        """
        self.seed = seed
        self.output_dim = output_dim

    def embedText(
        self, text: List[str], shared_feature_vector: __BuilderVector__, offset: int
    ) -> None:
        """Encodes a list of strings as an integer. This method is only called by TextBlock."""
        tokens = sorted(
            [
                mmh3.hash(string, signed=False) % self.output_dim + offset
                for string in text
            ]
        )

        last_char = ""
        unique = 0
        for t in tokens:
            if t != last_char:
                unique += 1
                last_char = t

        idxs = np.zeros((unique,))
        vals = np.zeros((unique,))

        last_char = ""
        i = -1
        for t in tokens:
            if t != last_char:
                i += 1
                last_char = t
                idxs[i] = t
            vals[i] += 1

        shared_feature_vector.addSparseFeatures(idxs, vals)

    def is_dense(self) -> bool:
        """True if the model produces dense features, False otherwise."""
        return False

    def feature_dim(self) -> int:
        """The dimension of the embedding produced by this model."""
        return self.output_dim
