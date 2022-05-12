#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <cstdint>
#include <vector>

// #include <dataset/src/utils/ExtendableVectors.h>

namespace thirdai::dataset {

/**
 * Declare here so we can make it a friend of
 * ExtendableVector.
 */
struct Block;
class ExtendableVectorTest;

/**
 * Extendable vector abstract class.
 * A vector representation that can be extended with a
 * new vector and can be converted into a BoltVector.
 *
 */
struct ExtendableVector {
  friend Block;
  friend ExtendableVectorTest;

 protected:
  /**
   * Extends the current vector by the given dimension.
   * Must be called exactly once per sample per block,
   * so to prevent erroneous use, we are making this a
   * protected method so it is only accessible to
   * derived classes and the Block abstract class.
   */
  virtual void extendByDim(uint32_t dim) = 0;

 public:
  /**
   * Sets a feature of the extension vector according to the
   * given index and value.
   */
  virtual void addExtensionSparseFeature(uint32_t index, float value) = 0;

  /**
   * Sets the next element of the dense extension vector to
   * the given value.
   */
  virtual void addExtensionDenseFeature(float value) = 0;

  /**
   * Given a sequence of possibly-repeating indices,
   * increment the features of the extension vector at
   * these indices. Repetitions are summed.
   */
  virtual void incrementExtensionAtIndices(std::vector<uint32_t>& indices,
                                           float inc) = 0;

  /**
   * Converts this vector to a BoltVector.
   */
  virtual bolt::BoltVector toBoltVector() = 0;
};

/**
 * Block abstract class.
 * A block accepts an input sample in the form of a sequence of strings
 * then encodes this sequence as a vector.
 */
struct Block {
  /**
   * Encodes a sequence of strings as a vector and concatenates the given
   * vector with this encoding.
   *
   * Arguments:
   * input_row: input sample; the sequence of strings to encoded.
   * vec: the vector to be concatenated with the vector
   *   encoding of input_row.
   */
  void extendVector(const std::vector<std::string>& input_row,
                    ExtendableVector& vec) {
    vec.extendByDim(featureDim());
    buildExtension(input_row, vec);
  }

  /**
   * Returns the dimension of the vector encoding.
   */
  virtual uint32_t featureDim() = 0;

  /**
   * True if the block produces dense features, False otherwise.
   */
  virtual bool isDense() = 0;

 protected:
  /**
   * Derived class-specific implementation of how input rows get
   * encoded (and what ends up in the extension vector).
   */
  virtual void buildExtension(const std::vector<std::string>& input_row,
                              ExtendableVector& vec) = 0;
};

}  // namespace thirdai::dataset