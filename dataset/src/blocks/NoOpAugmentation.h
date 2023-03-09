#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/Augmentation.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/InputTypes.h>
#include <memory>

namespace thirdai::dataset {

class NoOpAugmentation final : public Augmentation {
 public:
  NoOpAugmentation() {}

  std::vector<std::vector<BoltVector>> augment(
      std::vector<SegmentedFeatureVectorPtr>&& builders,
      ColumnarInputSample& sample) final {
    (void)sample;
    std::vector<std::vector<BoltVector>> vectors(builders.size());
    for (uint32_t builder_id = 0; builder_id < builders.size(); builder_id++) {
      vectors.at(builder_id) = {builders.at(builder_id)->toBoltVector()};
    }
    return vectors;
  }

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) final {
    (void)column_number_map;
  }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Augmentation>(this));
  }
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::NoOpAugmentation)