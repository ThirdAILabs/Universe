from typing import List, Callable
import numpy as np
from .text_embedding_model_interface import TextEmbeddingModel
from ..__utils__.builder_vectors import BuilderVector


class CustomDenseTextEmbedding(TextEmbeddingModel):
    """A dense text embedding model with a user-defined embedding function.
    Allows the user to extend the TextEmbeddingModel interface through
    composition. This class makes it easy to add new text embedding
    models or use existing third-party models.

    We have this class instead of passing an embedding function straight to
    TextBlock to prevent the TextBlock from having to distinguish between
    dense and sparse embeddings, but this may change in the near future.
    """

    def __init__(self, embed_fn: Callable[[List[str]], np.ndarray], out_dim: int):
        """Constructor.

        Arguments:
          embed_fn: function that takes in a list of strings and returns a dense
            embedding as a numpy array - the custom embedding function.
          out_dim: int - the dimension of the embedding returned by the custom
            function.
        """
        self.embed_fn = embed_fn
        self.out_dim = out_dim

    def is_dense(self) -> bool:
        """True if the model produces dense features, False otherwise."""
        return True

    def feature_dim(self) -> int:
        """The dimension of the embedding produced by this model."""
        return self.out_dim

    def embedText(
        self, text: List[str], shared_feature_vector: BuilderVector, offset: int
    ) -> None:
        """Encodes a list of strings as an integer. This method is only called by TextBlock."""
        shared_feature_vector.addDenseFeatures(offset, self.embed_fn(text))
