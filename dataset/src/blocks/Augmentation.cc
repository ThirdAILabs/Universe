#include "Augmentation.h"

namespace thirdai::dataset {

AugmentationList::AugmentationList(std::vector<AugmentationPtr> augmentations)
    : _augmentations(std::move(augmentations)) {}

Vectors AugmentationList::augment(Vectors&& vectors, ColumnarInputSample& input_sample) {
  for (auto& augmentation : _augmentations) {
    vectors = augmentation->augment(std::move(vectors), input_sample);
  }
  return vectors;
}

void AugmentationList::updateColumnNumbers(const ColumnNumberMap& column_number_map) {
  for (auto& augmentation : _augmentations) {
    augmentation->updateColumnNumbers(column_number_map);
  }
}

uint32_t AugmentationList::expectedNumColumns() const {
  uint32_t expectedNumColumns = 0;
  for (const auto& augmentation : _augmentations) {
    expectedNumColumns =
        std::max(expectedNumColumns, augmentation->expectedNumColumns());
  }
  return expectedNumColumns;
}

bool AugmentationList::isDense(uint32_t vector_index) const {
  bool is_dense = true;
  for (const auto& augmentation : _augmentations) {
    is_dense = is_dense && augmentation->isDense(vector_index);
  }
  return is_dense;
}
}  // namespace thirdai::dataset