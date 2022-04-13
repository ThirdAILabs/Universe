from typing import List
from ..utils.builder_vectors import __BuilderVector__


class TextEmbeddingModel:
    """Interface for text embedding models"""

    def embedText(
        self, text: List[str], shared_feature_vector: __BuilderVector__, offset: int
    ) -> None:
        """Encodes a list of strings as an integer. This method is only called by TextBlock."""
        return

    def is_dense(self) -> bool:
        """True if the model produces dense features, False otherwise."""
        return

    def feature_dim(self) -> int:
        """The dimension of the embedding produced by this model."""
        return
