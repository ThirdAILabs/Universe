#include "Dataset.h"
#include <stdexcept>

namespace thirdai::bolt::train {

Dataset convertDataset(dataset::BoltDataset&& dataset, uint32_t dim) {
  std::vector<nn::tensor::TensorPtr> tensors;

  for (uint32_t i = 0; i < dataset.numBatches(); i++) {
    tensors.push_back(
        nn::tensor::Tensor::convert(std::move(dataset.at(i)), dim));
  }

  return tensors;
}

}  // namespace thirdai::bolt::train