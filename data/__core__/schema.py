from ..__blocks__.block_interface import Block
from typing import List

class Schema:
  def __init__(self, input_feature_blocks: List[Block]=[], 
               target_feature_blocks: List[Block]=[]):
    self.input_feature_offsets = [0]
    for block in input_feature_blocks:
      self.input_feature_offsets.append(block.feature_dim() + self.input_feature_offsets[-1])
    self.input_feature_blocks = input_feature_blocks

    self.target_feature_offsets = [0]
    for block in input_feature_blocks:
      self.target_feature_offsets.append(block.feature_dim() + self.target_feature_offsets[-1])
    self.target_feature_blocks = target_feature_blocks
  
  def add_input_feature_block(self, block: Block) -> None:
    """Just like how tf.Sequential supports both initialization with a list of 
    models and add() methods.
    """
    next_offset = self.input_feature_offsets[-1] + block.feature_dim()
    self.input_feature_offsets.append(next_offset)
    self.input_feature_blocks.append(block)
  
  def add_target_feature_block(self, block: Block) -> None:
    """Same as above
    """
    next_offset = self.target_feature_offsets[-1] + block.feature_dim()
    self.target_feature_offsets.append(next_offset)
    self.target_feature_blocks.append(block)