from typing import List, Callable
import cytoolz as ct
from .block_interface import Block
from ..__utils__.builder_vectors import (
    BuilderVector,
    SparseBuilderVector,
    DenseBuilderVector,
)
from ..__models__.text_embedding_model_interface import TextEmbeddingModel


class TextBlock(Block):
    """A block for embedding a sample's raw textual features."""

    def __init__(
        self,
        column: int,
        embedding_model: TextEmbeddingModel,
        pipeline: List[Callable[[List[str]], List[str]]] = [],
    ):
        """Constructor.

        Arguments:
          column: int - given a sample as a row of raw features, column is the position of
            the text to be embedded within this row; "we want to embed the text in
            column n of the sample."
          embedding_model: TextEmbeddingModel - the model for encoding the text as vectors.
          pipeline: list of functions that accept and return a list of strings - a pipeline
            for preprocessing string tokens before they get encoded by the embedding model.
        """
        self.column = column
        self.embedding_model = embedding_model
        self.dense = embedding_model.is_dense()
        self.preprocess = lambda str_list: ct.pipe(str_list, *pipeline)

    def process(
        self,
        input_row: List[str],
        shared_feature_vector: BuilderVector = None,
        idx_offset=0,
    ) -> BuilderVector:
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

        Return value:
          A builder vector
        """
        if shared_feature_vector is None:
            idx_offset = 0
            shared_feature_vector = (
                DenseBuilderVector if self.dense else SparseBuilderVector
            )
        preprocessed_list_of_strings = self.preprocess([input_row[self.column]])
        self.embedding_model.embedText(
            preprocessed_list_of_strings, shared_feature_vector, idx_offset
        )
        return shared_feature_vector

    def feature_dim(self) -> int:
        """Returns the dimension of output vectors.
        This is needed when composing different features into a single vector.
        """
        return self.embedding_model.feature_dim()

    def is_dense(self) -> bool:
        """True if the block produces dense features, False otherwise.
        Follows the embedding model.
        """
        return self.embedding_model.is_dense()
