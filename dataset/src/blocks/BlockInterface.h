#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
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
// TODO(Geordie / Yash): it might make more sense to make
// percentage_significance unsigned and add a "correlation" field that is
// either positive or negative
struct Explanation {
  Explanation(uint32_t column_number, std::string keyword)
      : column_number(column_number), keyword(std::move(keyword)) {}

  Explanation(std::string column_name, std::string keyword)
      : column_number(0),
        keyword(std::move(keyword)),
        column_name(std::move(column_name)) {}

  Explanation(const ColumnIdentifier& column_identifier, std::string keyword)
      : keyword(std::move(keyword)) {
    if (column_identifier.hasName()) {
      column_name = column_identifier.name();
    }
    if (column_identifier.hasNumber()) {
      column_number = column_identifier.number();
    }
  }

  std::string toString() const {
    std::stringstream s;
    s << "column_name: \"" << column_name << "\" | keyword: \"" << keyword
      << "\" | percentage_significance: " << percentage_significance;
    return s.str();
  }

  uint32_t column_number;
  float percentage_significance = 0.0;
  // The following fields default to empty strings.
  std::string keyword;
  std::string column_name;
};

struct SegmentFeature {
  SegmentFeature(uint32_t segment_idx, uint32_t feature_idx)
      : segment_idx(segment_idx), feature_idx(feature_idx) {}
  uint32_t segment_idx;
  uint32_t feature_idx;
};

using IndexToSegmentFeatureMap = std::unordered_map<uint32_t, SegmentFeature>;

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

  virtual IndexToSegmentFeatureMap getIndexToSegmentFeatureMapImpl() = 0;

  const bool _store_index_to_segment_feature_map;

  IndexToSegmentFeatureMap _index_to_segment_feature;

 public:
  explicit SegmentedFeatureVector(bool store_segment_feature_map)
      : _store_index_to_segment_feature_map(store_segment_feature_map) {}

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

  IndexToSegmentFeatureMap getIndexToSegmentFeatureMap() {
    if (!_store_index_to_segment_feature_map) {
      throw std::invalid_argument(
          "[SegmentedFeatureVector::getSegmentFeatureMap] Attempted to get "
          "segment feature map when store_segment_feature_map is false.");
    }
    return getIndexToSegmentFeatureMapImpl();
  }

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
   */
  void addVectorSegment(ColumnarInputSample& input,
                        SegmentedFeatureVector& vec) {
    vec.addFeatureSegment(featureDim());
    buildSegment(input, vec);
  }

  /**
   * Updates the column numbers corresponding to each column name used by this
   * block. Throws an error if the block is not initialized with column names.
   * The column numbers allow the block to efficiently read from a tabular
   * dataset.
   */
  void updateColumnNumbers(const ColumnNumberMap& column_number_map) {
    for (auto* column_identifier : getColumnIdentifiers()) {
      column_identifier->updateColumnNumber(column_number_map);
    }
  }

  /**
   * Returns true if all of the current block's column identifiers have a column
   * number, returns false otherwise.
   */
  bool hasColumnNumbers() {
    auto column_identifiers = getColumnIdentifiers();
    if (column_identifiers.empty()) {
      return false;
    }
    // We don't have to go through all column identifiers because the
    // getConsistentColumnIdentifiers ensures consistency.
    return column_identifiers.front()->hasNumber();
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
  uint32_t computeExpectedNumColumns() {
    if (!hasColumnNumbers()) {
      return 0;
    }
    uint32_t expected_num_columns = 0;
    for (auto* column_identifier : getColumnIdentifiers()) {
      expected_num_columns =
          std::max(expected_num_columns, column_identifier->number() + 1);
    }
    return expected_num_columns;
  }

  /**
   * DO NOT CALL IN A PARALLEL REGION
   * Allows blocks to prepare for the incoming batch without being affected by
   * parallelism.
   *
   * One place where this is used is the UserCountHistoryBlock. There, we use
   * this method to determine whether it is safe to discard old counts.
   *
   * Avoid if possible, or minimize the amount of processing that happens here
   * since it may incur an overhead.
   */
  virtual void prepareForBatch(ColumnarInputBatch& incoming_batch) {
    (void)incoming_batch;
  }

  /**
   * For a given index, get the keyword which falls in that index when
   * building the segmented feature vector.
   *
   * Arguments:
   * index_within_block : index within the block so that we can get exact
   * keyword responsible.
   * input_row: the string_view of input string so that we process the
   * keywords when we call explainIndex method rather than storing that in
   * buildsegment , which may affect thread safety.
   *
   * Returns:
   * column number and keyword responsible for the given index from that
   * column.
   */
  virtual Explanation explainIndex(uint32_t index_within_block,
                                   ColumnarInputSample& input_row) = 0;

  virtual ~Block() = default;

 protected:
  /**
   * Derived class-specific implementation of how input rows get
   * encoded (and what ends up in the vector segment).
   *
   * WARNING: This function may be called in many threads simultaneously,
   * so it should be thread-safe or robust to data races. However, since this
   * function is always wrapped by a function that will catch exceptions
   * before they break the pragma omp loop, implementations should feel free to
   * throw exceptions.
   */
  virtual void buildSegment(ColumnarInputSample& input_row,
                            SegmentedFeatureVector& vec) = 0;

  /**
   * Gets all column identifiers used by the current block.
   */
  virtual std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() = 0;

 private:
  /**
   * All of a block's column identifiers must be either all initialized with a
   * name or all initialized with a number. Must be called by blocks during
   * construction to throw an error if this condition is not fulfilled.
   */
  std::vector<ColumnIdentifier*> getColumnIdentifiers() {
    auto column_identifiers = concreteBlockColumnIdentifiers();
    if (column_identifiers.empty()) {
      return column_identifiers;
    }

    auto* first_column_identifier = column_identifiers.front();
    for (const auto& column_identifier : column_identifiers) {
      if (!first_column_identifier->consistentWith(*column_identifier)) {
        throw std::invalid_argument(
            "ColumnIdentifiers are inconsistent; some have numbers/names while "
            "others don't.");
      }
    }
    return column_identifiers;
  }

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

}  // namespace thirdai::dataset