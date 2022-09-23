#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

/**
 * Declare here so we can make it a friend of
 * SegmentedFeatureVector.
 */
class Block;
class SegmentedFeatureVectorTest;
class CategoricalBlockTest;
class TextBlockTest;

using BlockPtr = std::shared_ptr<Block>;

/**
 * Helpful struct to keep all types of explanations required at one place.
 *
 * 1. percentage_significance : value which tells us how much this token is
 * responsible.
 * 2. column_number : column number corresponding to the responsible token.
 * 3. keyword : The main thing in our RCA which gives us exact
 * keyword is responsible for this.
 * 4. column_name : if the classifer has map we can return column_name also.
 */
struct Explanation {
  float percentage_significance;
  uint32_t column_number;
  std::string keyword;
  std::string column_name;
};

/**
 * Segmented feature vector abstract class.
 * A vector representation that can be extended with
 * feature segments and can be converted into a BoltVector.
 *
 * This is used when we want to compose features from various
 * feature blocks. Suppose we want an input vector that encodes
 * both text features and categorical features from raw data.
 * This data structure helps us create a vector that has one
 * block containing features extracted from raw text features,
 * and another segment containing features extracted from raw
 * categorical features.
 */
class SegmentedFeatureVector {
 public:
  friend Block;
  friend CategoricalBlockTest;
  friend TextBlockTest;
  friend SegmentedFeatureVectorTest;

 protected:
  /**
   * Adds a segment with the given dimension to the
   * current vector.
   *
   * This method is used by feature blocks to add
   * feature segments to a vector. Internally, this
   * method notifies the vector data structure to do
   * any relevant bookkeeping.
   *
   * This method must be called exactly once per
   * sample per block, so to prevent erroneous use,
   * we restrict access by making it a protected
   * method.
   */
  virtual void addFeatureSegment(uint32_t dim) = 0;

  /**
   * Returns a mapping of all of the vector's idx-value pairs.
   * Only used for testing as this can be very expensive
   * in dense vectors, so we restrict access by making
   * it a protected method.
   */
  virtual std::unordered_map<uint32_t, float> entries() = 0;

 public:
  /**
   * Increments the feature at the given index of the current vector segment
   * by a value.
   */
  virtual void addSparseFeatureToSegment(uint32_t index, float value) = 0;

  /**
   * Sets the next element of the dense vector segment to
   * the given value.
   */
  virtual void addDenseFeatureToSegment(float value) = 0;

  /**
   * Converts this vector to a BoltVector.
   */
  virtual BoltVector toBoltVector() = 0;

  virtual ~SegmentedFeatureVector() = default;
};

/**
 * Block abstract class.
 * A block accepts an input sample in the form of a sequence of strings
 * then encodes this sequence as a vector.
 */
class Block {
 public:
  /**
   * Encodes a sequence of strings as a vector and concatenates the given
   * vector with this encoding.
   *
   * Arguments:
   * input_row: input sample; the sequence of strings to encoded.
   * vec: the vector to be concatenated with the vector
   *   encoding of input_row.
   *
   * Returns:
   * exception_ptr: Since blocks can run in parallel in pragma
   * threads, they can't throw their own exceptions. To fail in a block, return
   * any exception_ptr and proceed with program execution without failing. The
   * error should then be caught.
   */
  std::exception_ptr addVectorSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) {
    vec.addFeatureSegment(featureDim());
    return buildSegment(input_row, vec);
  }

  /**
   * Returns the dimension of the vector encoding.
   */
  virtual uint32_t featureDim() const = 0;

  /**
   * True if the block produces dense features, False otherwise.
   */
  virtual bool isDense() const = 0;

  /**
   * Returns the minimum number of columns that the block expects
   * to see in each row of the dataset.
   */
  virtual uint32_t expectedNumColumns() const = 0;

  /**
   * For a given index, get the keyword which falls in that index when building
   * the segmented feature vector.
   *
   * Arguments:
   * index_within_block : index within the block so that we can get exact
   * keyword responsible.
   * input_row: the string_view of input string so that we process the
   * keywords when we call explainIndex method rather than storing that in
   * buildsegment , which may affect thread safety.
   *
   * Returns:
   * column number and keyword responsible for the given index from that column.
   */
  virtual Explanation explainIndex(
      uint32_t index_within_block,
      const std::vector<std::string_view>& input_row) const = 0;

  virtual ~Block() = default;

 protected:
  /**
   * Derived class-specific implementation of how input rows get
   * encoded (and what ends up in the vector segment).
   *
   * WARNING: This function may be called in many threads simultaneously,
   * so it should be thread-safe or robust to data races.
   */
  virtual std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) = 0;
};

}  // namespace thirdai::dataset