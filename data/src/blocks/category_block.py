from typing import List, Callable
import cytoolz as ct
from .block_interface import Block
from utils.builder_vectors import (
    __BuilderVector__,
    __SparseBuilderVector__,
    __DenseBuilderVector__,
)

class CategoryBlock(Block):
    """A block for embedding a sample's categorical features."""

    def __init__(
        self,
        column: int,
        dim: int,
    ):
        """Constructor.

        Arguments:
          column: int - given a sample as a row of raw features, column is the position of
            the categorical feature to be embedded within this row; "we want to embed the text in
            column n of the sample."
          dim: int - number of possible categories. 
        """
        self.column = column
        self.dim = dim
        self.dense = False

    def process(
        self,
        input_row: List[str],
        shared_feature_vector: __BuilderVector__,
        idx_offset=0,
    ) -> None:
        """Produces a vector out of textual information from the specified column
        of a sample.

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
        """
        category = int(input_row[self.column])
        shared_feature_vector.addSingleFeature(category % self.dim, 1.0)
        

    def feature_dim(self) -> int:
        """Returns the dimension of output vectors.
        This is needed when composing different features into a single vector.
        """
        return self.dim

    def is_dense(self) -> bool:
        """True if the block produces dense features, False otherwise.
        Follows the embedding model.
        """
        return self.dense
