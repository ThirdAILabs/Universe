#pragma once

#include <cereal/access.hpp>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <cstdint>
#include <memory>
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

  virtual bool isDense(uint32_t vector_index) const = 0;

  virtual ~Augmentation() = default;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using AugmentationPtr = std::shared_ptr<Augmentation>;

class AugmentationList final : public Augmentation {
 public:
  explicit AugmentationList(std::vector<AugmentationPtr> augmentations = {});

  Vectors augment(Vectors&& vectors, ColumnarInputSample& input_sample) final;

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) final;

  uint32_t expectedNumColumns() const final;

  bool isDense(uint32_t vector_index) const final;

  bool empty() const { return _augmentations.empty(); }

 private:
  std::vector<AugmentationPtr> _augmentations;

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::dataset