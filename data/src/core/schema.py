from ..blocks.block_interface import Block
from typing import Iterator, List, Tuple


class __BlockList__:
    """Wrapper class around a list of blocks that also handles feature offsetting,
    which ensures that the features extracted by different blocks occupy different
    parts of the composed feature vector.
    This prevents repeating code if Schema contains more than one list of blocks.
    """

    def __init__(self, blocks: List[Block] = []) -> None:
        """Constructor. Optionally takes in a list of blocks."""
        # self.offsets maps blocks to the starting positions of the corresponding
        # extracted feature in the composed vector;
        # self.offsets[i] = the starting position of the feature produced by
        # self.blocks[i].process()

        # The features produced by a block always starts where the features
        # produced by the previous block ends;
        # self.offsets[i] = self.offsets[i - 1] + dimension of self.blocks[i - 1]

        self.offsets = [0]
        for block in blocks:
            self.offsets.append(block.feature_dim() + self.offsets[-1])
        self.blocks = blocks
        # The collection of blocks produces dense features only if all blocks
        # produce dense features.
        # If any of the blocks produces a sparse feature, this collection of
        # blocks produces sparse features.
        self._is_dense = all([block.is_dense() for block in self.blocks])

    def add_block(self, block: Block) -> None:
        """A method to add blocks to this block list.
        This method facilitates a builder pattern invocation.
        """
        next_offset = self.offsets[-1] + block.feature_dim()
        self.offsets.append(next_offset)
        self.blocks.append(block)
        self._is_dense = self._is_dense and block.is_dense()

    def __len__(self) -> int:
        """Returns number of blocks."""
        return len(self.blocks)

    def __iter__(self) -> Iterator[Tuple[Block, int]]:
        """Iterates through (block, offset) pairs."""
        return iter(zip(self.blocks, self.offsets))

    def is_dense(self) -> bool:
        return self._is_dense


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

    def __init__(self, input_blocks: List[Block] = [], target_blocks: List[Block] = []):
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
        self.input_blocks = __BlockList__(input_blocks)
        self.target_blocks = __BlockList__(target_blocks)

    def add_input_block(self, block: Block) -> None:
        """A method to add features to the processed input vectors.
        This method facilitates a builder pattern invocation.
        """
        self.input_blocks.add_block(block)
        return self  # Return self so we can chain method calls

    def add_target_block(self, block: Block) -> None:
        """A method to add features to the processed target vectors.
        This method facilitates a builder pattern invocation.
        """
        self.target_blocks.add_block(block)
        return self  # Return self so we can chain method calls
