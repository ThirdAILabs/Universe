from typing import List
from thirdai._thirdai.dataset_internal import Block
from typing_extensions import Self


class Schema:
    """Identifies the raw features to be processed in each sample and how to
    process them.

    The schema identifies the features of both the input and target vectors.
    Input vectors are vectors that are passed as input into a downstream
    machine learning model while target vectors are what the model must
    learn to predict given the input vectors.

    The order of features in the generated vectors is the same as the order
    of the corresponding blocks.

    Example usage 1:
      product_name = TextBlock(column=0)
      sales_volume = NumberBlock(column=1)
      color = CategoryBlock(column=2)

      schema = Schema(
        input_blocks=[product_name, color],
        target_blocks=[sales_volume]
      )

    This means:
    - For each sample, we produce an input vector that encodes textual
      information from column 0 and categorical information from column 2 of
      the sample.
    - For each sample, we also produce a target vector that encodes
      numerical information from column 1 of the sample.
    - Suppose TextBlock produces a 1000-dimensional embedding, NumberBlock
      produces a 1-dimensional embedding, and CategoryBlock produces a
      10-dimensional embedding.

      This means the first 1000 dimensions of the generated input vector
      encode the product name while the next 10 dimensions encode product
      color, adding up to a 1010-dimensional input vector.
      The target vector encodes sales volume in 1 dimension.
    """

    def __init__(
        self, input_blocks: List[Block] = [], target_blocks: List[Block] = []
    ) -> None:
        """Constructor.

        Arguments:
          input_blocks: list of Blocks - identifies how a sample's raw
            features are encoded into an input vector for a downstream
            machine learning model.
          target_blocks: list of Blocks - identifies how a sample's raw
            features are encoded into a target vector for a downstream
            machine learning model.

        Arguments can be omitted in exchange for a builder pattern
        invocation.
        """
        self._input_blocks = []
        self._input_dim = 0
        self._target_blocks = []
        self._target_dim = 0

        for block in input_blocks:
            self.add_input_block(block)
        for block in target_blocks:
            self.add_target_block(block)

    def add_input_block(self, block: Block) -> Self:
        """Adds a feature to the processed input vectors.
        This method facilitates a builder pattern invocation.
        """
        self._input_blocks.append(block)
        self._input_dim += block.feature_dim()
        return self  # Return self so we can chain method calls

    def add_target_block(self, block: Block) -> Self:
        """Adds a feature to the processed target vectors.
        This method facilitates a builder pattern invocation.
        """
        self._target_blocks.append(block)
        self._target_dim += block.feature_dim()
        return self  # Return self so we can chain method calls

    def input_dim(self) -> int:
        """Returns dimension of input vectors."""
        return self._input_dim

    def target_dim(self) -> int:
        """Returns dimension of target vectors."""
        return self._target_dim
