#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>

namespace thirdai::bolt {

using Dataset = std::vector<TensorList>;

using LabeledDataset = std::pair<Dataset, Dataset>;

Dataset convertDatasets(const std::vector<dataset::BoltDatasetPtr>& datasets,
                        std::vector<uint32_t> dims, bool copy = true);

Dataset convertDataset(const dataset::BoltDatasetPtr& dataset, uint32_t dim,
                       bool copy = true);

TensorList convertBatch(std::vector<BoltBatch>&& batches,
                        const std::vector<uint32_t>& dims);

TensorList convertVectors(std::vector<BoltVector>&& vectors,
                          const std::vector<uint32_t>& dims);

}  // namespace thirdai::bolt