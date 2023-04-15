#include "Dataset.h"
#include <bolt/src/nn/tensor/Tensor.h>
#include <exception>
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

  std::exception_ptr err;

#pragma omp parallel for default(none) \
    shared(num_batches, datasets, batches, dims, err)
  for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    for (uint32_t dataset_idx = 0; dataset_idx < datasets.size();
         dataset_idx++) {
      try {
        batches.at(batch_idx).push_back(nn::tensor::Tensor::convert(
            datasets[dataset_idx]->at(batch_idx), dims.at(dataset_idx)));
      } catch (...) {
#pragma omp critical
        err = std::current_exception();
      }
    }
  }

  if (err) {
    std::rethrow_exception(err);
  }

  return batches;
}

Dataset convertDataset(const dataset::BoltDatasetPtr& dataset, uint32_t dim) {
  std::vector<dataset::BoltDatasetPtr> datasets = {dataset};
  return convertDatasets({std::move(datasets)}, {dim});
}

nn::tensor::TensorList convertBatch(const std::vector<BoltBatch>& batches,
                                    const std::vector<uint32_t>& dims) {
  if (dims.size() != batches.size()) {
    throw std::invalid_argument(
        "Expected number of dimensions to match the number of batches.");
  }

  nn::tensor::TensorList tensors;
  for (uint32_t i = 0; i < batches.size(); i++) {
    tensors.push_back(nn::tensor::Tensor::convert(batches[i], dims[i]));
  }

  return tensors;
}

nn::tensor::TensorList convertVectors(const std::vector<BoltVector>& vectors,
                                      const std::vector<uint32_t>& dims) {
  if (dims.size() != vectors.size()) {
    throw std::invalid_argument(
        "Expected number of dimensions to match the number of vectors.");
  }

  nn::tensor::TensorList tensors;
  for (uint32_t i = 0; i < vectors.size(); i++) {
    tensors.push_back(nn::tensor::Tensor::convert(vectors[i], dims[i]));
  }

  return tensors;
}

}  // namespace thirdai::bolt::train