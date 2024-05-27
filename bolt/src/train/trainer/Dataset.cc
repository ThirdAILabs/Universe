#include "Dataset.h"
#include <bolt/src/nn/tensor/Tensor.h>
#include <exception>
#include <stdexcept>
#include <string>

namespace thirdai::bolt {

Dataset convertDatasets(const std::vector<dataset::BoltDatasetPtr>& datasets,
                        std::vector<uint32_t> dims, bool copy) {
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

  std::vector<TensorList> batches(num_batches);

  std::exception_ptr err;

#pragma omp parallel for default(none) \
    shared(num_batches, datasets, batches, dims, copy, err)
  for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    for (uint32_t dataset_idx = 0; dataset_idx < datasets.size();
         dataset_idx++) {
      try {
        if (copy) {
          batches.at(batch_idx).push_back(Tensor::copy(
              datasets[dataset_idx]->at(batch_idx), dims.at(dataset_idx)));
        } else {
          batches.at(batch_idx).push_back(
              Tensor::convert(std::move(datasets[dataset_idx]->at(batch_idx)),
                              dims.at(dataset_idx)));
        }
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

Dataset convertDataset(const dataset::BoltDatasetPtr& dataset, uint32_t dim,
                       bool copy) {
  std::vector<dataset::BoltDatasetPtr> datasets = {dataset};
  return convertDatasets({std::move(datasets)}, {dim}, copy);
}

TensorList convertBatch(std::vector<BoltBatch>&& batches,
                        const std::vector<uint32_t>& dims) {
  if (dims.size() != batches.size()) {
    throw std::invalid_argument(
        "Expected number of dimensions to match the number of batches.");
  }

  TensorList tensors;
  for (uint32_t i = 0; i < batches.size(); i++) {
    tensors.push_back(Tensor::convert(std::move(batches[i]), dims[i]));
  }

  return tensors;
}

TensorList convertVectors(std::vector<BoltVector>&& vectors,
                          const std::vector<uint32_t>& dims) {
  if (dims.size() != vectors.size()) {
    throw std::invalid_argument(
        "Expected number of dimensions to match the number of vectors.");
  }

  TensorList tensors;
  for (uint32_t i = 0; i < vectors.size(); i++) {
    tensors.push_back(Tensor::convert(std::move(vectors[i]), dims[i]));
  }

  return tensors;
}

}  // namespace thirdai::bolt