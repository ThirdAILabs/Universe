#pragma once

#include <bolt/src/graph/tests/TestDatasetGenerators.h>
#include <bolt/src/train/trainer/Dataset.h>

namespace thirdai::bolt::nn::tests {

inline train::LabeledDataset getLabeledDataset(uint32_t n_classes,
                                               uint32_t n_batches,
                                               uint32_t batch_size,
                                               bool sparse = false) {
  auto [data, labels] =
      thirdai::bolt::tests::TestDatasetGenerators::generateSimpleVectorDataset(
          /* n_classes= */ n_classes, /* n_batches= */ n_batches,
          /* batch_size= */ batch_size, /* noisy_dataset= */ false,
          /* sparse= */ sparse);

  return {train::convertDataset(data, n_classes),
          train::convertDataset(labels, n_classes)};
}

}  // namespace thirdai::bolt::nn::tests