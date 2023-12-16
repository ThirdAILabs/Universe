#pragma once

#include <cereal/access.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/Augmentation.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <utils/text/StringManipulation.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>

namespace thirdai::dataset {

class RecurrenceAugmentation final : public Augmentation {
 public:
  static constexpr const char* EOS = "$EOS$";

  explicit RecurrenceAugmentation(ColumnIdentifier sequence_column,
                                  char delimiter, uint32_t max_recurrence,
                                  uint32_t vocab_size,
                                  uint32_t input_vector_index,
                                  uint32_t label_vector_index);

  std::vector<std::vector<BoltVector>> augment(
      std::vector<SegmentedFeatureVectorPtr>&& builders,
      ColumnarInputSample& input_sample) final;

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) final {
    _sequence_column.updateColumnNumber(column_number_map);
  }

  BlockPtr inputBlock() {
    return std::make_shared<PlaceholderBlock>(
        /* name= */ "RecurrenceInput",
        /* dim= */ _vocab.maxSize().value() * _max_recurrence,
        /* dense= */ false, /* column_identifier= */ _sequence_column);
  }

  BlockPtr labelBlock() {
    return std::make_shared<PlaceholderBlock>(
        /* name= */ "RecurrenceLabel",
        /* dim= */ _vocab.maxSize().value() * _max_recurrence,
        /* dense= */ false, /* column_identifiers= */ _sequence_column);
  }

  uint32_t elementIdAtStep(const BoltVector& output, uint32_t step);

  std::string elementString(uint32_t element_id);

  bool isEOS(uint32_t element_id);

  static auto make(ColumnIdentifier sequence_column, char delimiter,
                   uint32_t max_recurrence, uint32_t vocab_size,
                   uint32_t input_vector_index, uint32_t label_vector_index) {
    return std::make_shared<RecurrenceAugmentation>(
        std::move(sequence_column), delimiter, max_recurrence, vocab_size,
        input_vector_index, label_vector_index);
  }

 private:
  std::vector<std::string> sequence(ColumnarInputSample& input_sample) const;

  std::vector<uint32_t> elementIds(const std::vector<std::string>& sequence);

  static std::vector<BoltVector> augmentInputVectors(
      SegmentedFeatureVector& builder, std::vector<uint32_t> elements);

  static std::vector<BoltVector> augmentLabelVectors(
      SegmentedFeatureVector& builder, std::vector<uint32_t> elements);

  static std::vector<BoltVector> replicateOtherVectors(
      SegmentedFeatureVector& builder, uint32_t size);

  ColumnIdentifier _sequence_column;
  char _delimiter;
  uint32_t _max_recurrence;
  uint32_t _input_vector_index;
  uint32_t _label_vector_index;
  ThreadSafeVocabulary _vocab;

  // Private default constructor for cereal.
  RecurrenceAugmentation() : _vocab(/* vocab_size= */ 0) {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);
};

using RecurrenceAugmentationPtr = std::shared_ptr<RecurrenceAugmentation>;

}  // namespace thirdai::dataset