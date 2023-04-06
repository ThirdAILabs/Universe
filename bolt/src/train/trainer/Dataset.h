#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <dataset/src/Datasets.h>

namespace thirdai::bolt::train {

using Dataset = std::vector<nn::tensor::TensorList>;

using LabeledDataset = std::pair<Dataset, Dataset>;

Dataset convertDatasets(const std::vector<dataset::BoltDatasetPtr>& datasets,
                        std::vector<uint32_t> dims);

Dataset convertDataset(const dataset::BoltDatasetPtr& dataset, uint32_t dim);

}  // namespace thirdai::bolt::train