#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <hashing/src/HashUtils.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <utils/StringManipulation.h>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace thirdai::dataset {

/**
 * A block that encodes text (e.g. sentences / paragraphs).
 */
class SequenceBlock : public Block {
 public:
  explicit SequenceBlock(ColumnIdentifier col, char delimiter, uint32_t dim);

  uint32_t featureDim() const final;

  bool isDense() const final;

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final;

  static auto make(ColumnIdentifier col, char delimiter,
                   uint32_t dim = TokenEncoding::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<SequenceBlock>(std::move(col), delimiter, dim);
  }

 protected:
  std::exception_ptr buildSegment(ColumnarInputSample& input,
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

class SequenceTargetBlock : public Block {
 public:
  // Vocabulary size includes EOS character if relevant
  SequenceTargetBlock(ColumnIdentifier target_col, ColumnIdentifier step_col,
                      uint32_t max_steps, uint32_t vocabulary_size);

  uint32_t featureDim() const final;

  bool isDense() const final;

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final;

  static auto make(ColumnIdentifier target_col, ColumnIdentifier step_col,
                   uint32_t max_steps, uint32_t vocabulary_size) {
    return std::make_shared<SequenceTargetBlock>(
        std::move(target_col), std::move(step_col), max_steps, vocabulary_size);
  }

  std::string className(uint32_t label_id);

  std::string classNameAtStep(const BoltVector& activations, uint32_t step);

 protected:
  std::exception_ptr buildSegment(ColumnarInputSample& input,
                                  SegmentedFeatureVector& vec) final;

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final;

 private:
  // Constructor for cereal.
  SequenceTargetBlock() : _vocabulary(0) {}

  ColumnIdentifier _target_col;
  ColumnIdentifier _step_col;
  uint32_t _max_steps;
  ThreadSafeVocabulary _vocabulary;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _target_col, _step_col, _max_steps,
            _vocabulary);
  }
};

using SequenceTargetBlockPtr = std::shared_ptr<SequenceTargetBlock>;

}  // namespace thirdai::dataset
