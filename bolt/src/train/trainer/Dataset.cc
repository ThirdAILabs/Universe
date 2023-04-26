#include "Dataset.h"
#include <stdexcept>
#include <string>

namespace thirdai::bolt::train {

Dataset convertDatasets(const std::vector<dataset::BoltDatasetPtr>& datasets,
                        std::vector<uint32_t> dims) {
  if (datasets.empty()) {
    return {};
  }

  if (dims.size() != datasets.size()) {
    throw std::invalid_argument(
        "Expected number of dimensions to match the number of datasets, but "
        "found " +
        std::to_string(dims.size()) + " dims and " +
        std::to_string(datasets.size()) + " datasets.");
  }

  uint32_t num_batches = datasets.front()->numBatches();
  for (const auto& dataset : datasets) {
    if (dataset->numBatches() != num_batches) {
      throw std::invalid_argument(
          "Expected all datasets to have the same number of batches for "
          "conversion.");
    }
  }

  std::vector<nn::tensor::TensorList> batches(num_batches);

#pragma omp parallel for default(none) \
    shared(num_batches, datasets, batches, dims)
  for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    for (uint32_t dataset_idx = 0; dataset_idx < datasets.size();
         dataset_idx++) {
      batches.at(batch_idx).push_back(nn::tensor::Tensor::convert(
          datasets[dataset_idx]->at(batch_idx), dims.at(dataset_idx)));
    }
  }

  return batches;
}

Dataset convertDatasetsForModelDims(
    const std::vector<dataset::BoltDatasetPtr>& datasets,
    const std::vector<nn::tensor::Dims>& input_dims) {
  std::vector<uint32_t> two_d_dims;
  for (const auto& dims : input_dims) {
    if (dims.size() != 1) {
      throw std::invalid_argument(
          "Can only convert bolt batches to 2D tensors but received tensor "
          "with dimensions " +
          nn::tensor::toString(dims) + ".");
    }
    two_d_dims.push_back(dims.back());
  }

  return convertDatasets(datasets, two_d_dims);
}

Dataset convertDataset(const dataset::BoltDatasetPtr& dataset, uint32_t dim) {
  std::vector<dataset::BoltDatasetPtr> datasets = {dataset};
  return convertDatasets(datasets, {dim});
}

Dataset convertDatasetForModelDim(const dataset::BoltDatasetPtr& dataset,
                                  nn::tensor::Dims dim) {
  std::vector<dataset::BoltDatasetPtr> datasets = {dataset};
  return convertDatasets(datasets, {std::move(dim)});
}

}  // namespace thirdai::bolt::train