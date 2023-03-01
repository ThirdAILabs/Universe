#pragma once

#include <cereal/access.hpp>
#include <dataset/src/blocks/Augmentation.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <utils/StringManipulation.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>

namespace thirdai::dataset {

class RecurrenceAugmentation final : public Augmentation {
 public:
  static constexpr auto EOS = "$EOS$";
  static constexpr size_t EOS_SIZE = 5;

  explicit RecurrenceAugmentation(ColumnIdentifier sequence_column,
                                  char delimiter, uint32_t max_recurrence,
                                  uint32_t vocab_size,
                                  uint32_t input_vector_index,
                                  uint32_t label_vector_index);

  void prepareForBatch(ColumnarInputBatch& incoming_batch) final;

  Vectors augment(Vectors&& vectors, ColumnarInputSample& input_sample) final;

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) final {
    _sequence_column.updateColumnNumber(column_number_map);
  }

  uint32_t expectedNumColumns() const final;

  bool isDense(uint32_t vector_index) const final {
    (void)vector_index;
    return false;
  }

  uint32_t featureDim(uint32_t vector_index) const final {
    if (vector_index == _in_progress_vector_index ||
        vector_index == _label_vector_index) {
      return _vocab.vocabSize() * _max_recurrence;
    }
    return 0;
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
  std::vector<std::string_view> sequence(
      ColumnarInputSample& input_sample) const;

  std::vector<uint32_t> elementIds(
      const std::vector<std::string_view>& sequence);

  Vectors augmentEach(SampleVector&& vectors,
                      const std::vector<uint32_t>& element_ids) const;

  static Vectors multiply(SampleVector&& vectors, uint32_t times);

  static SampleVector clone(const SampleVector& vectors);

  void addInProgressFeatures(SampleVector& vectors,
                             const std::vector<uint32_t>& element_ids,
                             uint32_t step) const;

  void addLabelFeatures(SampleVector& vectors,
                        const std::vector<uint32_t>& element_ids,
                        uint32_t step) const;

  ColumnIdentifier _sequence_column;
  char _delimiter;
  uint32_t _max_recurrence;
  uint32_t _in_progress_vector_index;
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