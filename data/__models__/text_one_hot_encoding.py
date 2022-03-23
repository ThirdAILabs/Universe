from typing import List
import mmh3
from .text_embedding_model_interface import TextEmbeddingModel
from ..__utils__.builder_vectors import BuilderVector


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
        self, text: List[str], shared_feature_vector: BuilderVector, offset: int
    ) -> None:
        """Encodes a list of strings as an integer. This method is only called by TextBlock."""
        for string in text:
            hash = mmh3.hash(string)
            shared_feature_vector.addSingleFeature(hash % self.output_dim + offset, 1.0)

    def is_dense(self) -> bool:
        """True if the model produces dense features, False otherwise."""
        return False

    def feature_dim(self) -> int:
        """The dimension of the embedding produced by this model."""
        return self.output_dim
