#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <dataset/src/Datasets.h>

namespace thirdai::bolt::train {

using Dataset = std::vector<nn::tensor::TensorPtr>;

using LabeledDataset = std::pair<Dataset, Dataset>;

void verifyNumBatchesMatch(const LabeledDataset& data);

Dataset convertDataset(dataset::BoltDataset&& dataset, uint32_t dim);

}  // namespace thirdai::bolt::train