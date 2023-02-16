#include "Dataset.h"
#include <dataset/src/Datasets.h>
#include <stdexcept>

namespace thirdai::bolt::train {

Dataset convertDatasets(std::vector<dataset::BoltDatasetPtr>&& datasets,
                        std::vector<uint32_t> dims) {
  if (datasets.empty()) {
    return {};
  }

  if (dims.size() != datasets.size()) {
    throw std::invalid_argument(
        "Expected number of dimensions to match the number of datasets.");
  }

  uint32_t num_batches = datasets.front()->numBatches();
  for (const auto& dataset : datasets) {
    if (dataset->numBatches() != num_batches) {
      throw std::invalid_argument(
          "Expected all datasets to have the same number of batches for "
          "conversion.");
    }
  }

  std::vector<nn::tensor::TensorList> tensors(num_batches);

#pragma omp parallel for default(none) \
    shared(num_batches, datasets, tensors, dims)
  for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    for (uint32_t dataset_idx = 0; dataset_idx < datasets.size();
         dataset_idx++) {
      tensors.at(batch_idx).push_back(nn::tensor::Tensor::convert(
          std::move(datasets[dataset_idx]->at(batch_idx)),
          dims.at(dataset_idx)));
    }
  }

  return tensors;
}

Dataset convertDataset(dataset::BoltDatasetPtr&& dataset, uint32_t dim) {
  std::vector<dataset::BoltDatasetPtr> datasets = {dataset};
  return convertDatasets(std::move(datasets), {dim});
}

}  // namespace thirdai::bolt::train