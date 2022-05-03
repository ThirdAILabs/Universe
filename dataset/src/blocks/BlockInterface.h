#pragma once

#include <string_view>
#include <dataset/src/BuilderVectors.h>

namespace thirdai::dataset {

/**
 * Block interface.
 * A block encodes a sample's raw features as a vector.
 * Concerete implementations of this interface handles 
 * specific types of raw features, e.g. text, category, 
 * number, timestamp.
 */
struct Block {

  /**
   * Extracts features from input row and adds it to shared feature vector.
   *
   * Arguments:
   * input_row: a list of columns for a single row.
   * shared_feature_vector: a vector that is shared among all blocks operating on
   *   a particular row. This make it easier for the pipeline object to
   *   concatenate the features produced by each block. 
   * idx_offset: the offset to shift the feature indices by if the preceeding
   *   section of the output vector is occupied by other features.
   */
  virtual void process(std::vector<std::string_view>& input_row, BuilderVector& shared_feature_vector, uint32_t idx_offset) = 0;

  /**
   * Returns the dimension of extracted features.
   * This is needed when composing different features into a single vector.
   */
  virtual uint32_t featureDim() = 0;

  /**
   * True if the block produces dense features, False otherwise.
   */
  virtual bool isDense() = 0;
};

} // namespace thirdai::dataset