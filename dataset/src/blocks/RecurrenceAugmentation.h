#pragma once

#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <utils/StringManipulation.h>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace thirdai::dataset {

using SampleVector = std::vector<std::shared_ptr<SegmentedFeatureVector>>;

using Vectors = std::vector<SampleVector>;

class Augmentation {
 public:
  virtual Vectors augment(Vectors&& vectors,
                          ColumnarInputSample& input_sample) = 0;

  virtual void updateColumnNumbers(
      const ColumnNumberMap& column_number_map) = 0;

  virtual uint32_t expectedNumColumns() const = 0;

  virtual bool isDense(uint32_t vector_index) = 0;

  virtual ~Augmentation() = default;
};

using AugmentationPtr = std::shared_ptr<Augmentation>;

class AugmentationList final : public Augmentation {
 public:
  explicit AugmentationList(std::vector<AugmentationPtr> augmentations)
      : _augmentations(std::move(augmentations)) {}

  Vectors augment(Vectors&& vectors, ColumnarInputSample& input_sample) final {
    for (auto& augmentation : _augmentations) {
      vectors = augmentation->augment(std::move(vectors), input_sample);
    }
    return vectors;
  }

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) final {
    for (auto& augmentation : _augmentations) {
      augmentation->updateColumnNumbers(column_number_map);
    }
  }

  uint32_t expectedNumColumns() const final {
    uint32_t expectedNumColumns = 0;
    for (const auto& augmentation : _augmentations) {
      expectedNumColumns =
          std::max(expectedNumColumns, augmentation->expectedNumColumns());
    }
    return expectedNumColumns;
  }

  bool isDense(uint32_t vector_index) final {
    bool is_dense = true;
    for (const auto& augmentation : _augmentations) {
      is_dense = is_dense && augmentation->isDense(vector_index);
    }
    return is_dense;
  }

 private:
  std::vector<AugmentationPtr> _augmentations;
};

class RecurrenceAugmentation final : public Augmentation {
 public:
  static constexpr auto EOS = "$EOS$";
  static constexpr size_t EOS_SIZE = 5;

  explicit RecurrenceAugmentation(ColumnIdentifier sequence_column,
                                  uint32_t max_recurrence, uint32_t vocab_size,
                                  uint32_t input_vector_index,
                                  uint32_t label_vector_index);

  Vectors augment(Vectors&& vectors, ColumnarInputSample& input_sample) final;

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) final {
    _sequence_column.updateColumnNumber(column_number_map);
  }

  uint32_t expectedNumColumns() const final {
    return _sequence_column.number() + 1;
  }

  bool isDense(uint32_t vector_index) final {
    (void)vector_index;
    return false;
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
  uint32_t _max_recurrence;
  uint32_t _vocab_size_with_eos;
  uint32_t _in_progress_vector_index;
  uint32_t _label_vector_index;
  ThreadSafeVocabulary _vocab;
};

}  // namespace thirdai::dataset