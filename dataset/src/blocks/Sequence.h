#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <hashing/src/HashUtils.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <utils/text/StringManipulation.h>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace thirdai::dataset {

/**
 * A block that encodes an ordered sequence of elements delimited by a
 * character. The encoded vector has disjoint ranges for each position in the
 * sequence.
 *
 * Given a sequence "a b c d", this block will create a vector that represents
 * {a_1, b_2, c_3, d_4}.
 */
class SequenceBlock : public Block {
 public:
  explicit SequenceBlock(ColumnIdentifier col, char delimiter, uint32_t dim);

  uint32_t featureDim() const final;

  bool isDense() const final;

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final;

  static auto make(ColumnIdentifier col, char delimiter,
                   uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<SequenceBlock>(std::move(col), delimiter, dim);
  }

 protected:
  void buildSegment(ColumnarInputSample& input,
                    SegmentedFeatureVector& vec) final;

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final;

 private:
  // Constructor for cereal.
  SequenceBlock() {}

  uint32_t sequenceHash(std::string_view element, uint32_t pos) const;

  ColumnIdentifier _col;
  char _delimiter;
  uint32_t _dim;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _col, _delimiter, _dim);
  }
};

using SequenceBlockPtr = std::shared_ptr<SequenceBlock>;

}  // namespace thirdai::dataset
