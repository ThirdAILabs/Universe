from utils.builder_vectors import __BuilderVector__
from typing import List


class Block:
    """Block interface.
    A block encodes a sample's raw features as a vector.

    Concerete implementations of this interface handles specific types of raw
    features, e.g. text, category, number, timestamp.
    """

    def process(
        self,
        input_row: List[str],  # TODO(Geordie): Find a way to support other
        # representations of input rows.
        shared_feature_vector: __BuilderVector__,
        idx_offset: int = 0,
    ) -> None:
        """Extracts features from input row and adds it to shared feature vector.

        Arguments:
          input_row: a list of columns for a single row.
          shared_feature_vector: a vector that is shared among all blocks operating on
            a particular row. This make it easier for the pipeline object to
            concatenate the features produced by each block.
          idx_offset: the offset to shift the feature indices by if the preceeding
            section of the output vector is occupied by other features.

        """
        return

    def featureDim(self) -> int:
        """Returns the dimension of extracted features.
        This is needed when composing different features into a single vector.
        """
        return

    def isDense(self) -> bool:
        """True if the block produces dense features, False otherwise."""
        return
