#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>

namespace thirdai::bolt::train {

using Dataset = std::vector<nn::tensor::TensorList>;

using LabeledDataset = std::pair<Dataset, Dataset>;

Dataset convertDatasets(const std::vector<dataset::BoltDatasetPtr>& datasets,
                        std::vector<uint32_t> dims, bool copy = true);

Dataset convertDataset(const dataset::BoltDatasetPtr& dataset, uint32_t dim,
                       bool copy = true);

nn::tensor::TensorList convertBatch(std::vector<BoltBatch>&& batches,
                                    const std::vector<uint32_t>& dims);

nn::tensor::TensorList convertVectors(std::vector<BoltVector>&& vectors,
                                      const std::vector<uint32_t>& dims);

nn::tensor::Dims expect2dDims(const std::vector<nn::tensor::Dims>& dims_nd) ;

}  // namespace thirdai::bolt::train