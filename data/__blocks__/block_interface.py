from ..__utils__.builder_vectors import BuilderVector
from typing import List


class Block:
    """Block interface.
    A block encodes a sample's raw features as a vector.

    Concerete implementations of this interface handles specific types of raw
    features, e.g. text, category, number, timestamp.
    """

    def process(
        self,
        input_row: List[str],
        shared_feature_vector: BuilderVector = None,
        idx_offset: int = 0,
    ) -> BuilderVector:
        """The block can return either dense or sparse features, depending on
        what is best for the feature that it produces.

        Arguments:
          input_row: a list of raw features from a single sample.
          shared_feature_vector: a vector that is shared among all blocks operating on
            a particular row. This make it easier for the dataset object to
            concatenate the features produced by each block. If not provided, the
            method creates its own vector, which could be sparse or dense, whatever
            is best suited for the feature it produces.
          idx_offset: the offset to shift the feature indices by if the preceeding
            section of the output vector is occupied by other features, only needed
            if shared_feature_vector is supplied.

        Return value:
          A builder vector
        """
        return

    def feature_dim(self) -> int:
        """Returns the dimension of output vectors.
        This is needed when composing different features into a single vector.
        """
        return

    def is_dense(self) -> bool:
        """True if the block produces dense features, False otherwise."""
        return
