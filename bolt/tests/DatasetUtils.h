#pragma once

#include <bolt/src/graph/tests/TestDatasetGenerators.h>
#include <bolt/src/train/trainer/Dataset.h>

namespace thirdai::bolt::nn::tests {

inline train::LabeledDataset getDataset(uint32_t n_classes, uint32_t n_batches,
                                        uint32_t batch_size) {
  auto [data, labels] =
      thirdai::bolt::tests::TestDatasetGenerators::generateSimpleVectorDataset(
          /* n_classes= */ n_classes, /* n_batches= */ n_batches,
          /* batch_size= */ batch_size, /* noisy_dataset= */ false);

  return {train::convertDataset(std::move(*data), n_classes),
          train::convertDataset(std::move(*labels), n_classes)};
}

}  // namespace thirdai::bolt::nn::tests