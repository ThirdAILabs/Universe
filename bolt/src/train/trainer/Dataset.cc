#include "Dataset.h"
#include <stdexcept>

namespace thirdai::bolt::train {

void verifyNumBatchesMatch(const LabeledDataset& data) {
  if (data.first.size() != data.second.size()) {
    throw std::invalid_argument(
        "Data and labels must have same number of batches.");
  }
}

Dataset convertDataset(dataset::BoltDataset&& dataset, uint32_t dim) {
  std::vector<nn::tensor::TensorPtr> tensors;

  for (uint32_t i = 0; i < dataset.numBatches(); i++) {
    tensors.push_back(
        nn::tensor::Tensor::convert(std::move(dataset.at(i)), dim));
  }

  return tensors;
}

}  // namespace thirdai::bolt::train