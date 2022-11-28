#include "DatasetContext.h"
#include <bolt/src/graph/nodes/Input.h>

namespace thirdai::bolt {

DatasetContext::DatasetContext(std::vector<dataset::BoltDatasetPtr> data,
                               dataset::BoltDatasetPtr labels)
    : _data(std::move(data)), _labels(std::move(labels)) {
  _all_dag_datasets.insert(_all_dag_datasets.end(), _data.begin(), _data.end());

  if (_labels) {
    _all_dag_datasets.push_back(_labels);
  }
  if (_all_dag_datasets.empty()) {
    throw std::invalid_argument(
        "Must pass in at least one dataset, but found 0.");
  }

  verifyDatasetLens(_all_dag_datasets);
  verifyBatchSizes(_all_dag_datasets);
}

void DatasetContext::setInputs(uint64_t batch_idx,
                               const std::vector<InputPtr>& inputs) {
  for (uint32_t i = 0; i < inputs.size(); i++) {
    inputs[i]->setInputs(&_data[i]->at(batch_idx));
  }
}

void DatasetContext::verifyBatchSizes(
    const dataset::DatasetBaseList& datasets) {
  uint64_t first_batch_size = datasets.front()->batchSize();
  for (const auto& dataset : datasets) {
    if (dataset->batchSize() != first_batch_size) {
      throw std::invalid_argument(
          "All datasets must have the same batch size, "
          "but found " +
          std::to_string(first_batch_size) +
          " for one dataset's batch size and " +
          std::to_string(dataset->batchSize()) + " for another");
    }
  }
}

void DatasetContext::verifyDatasetLens(
    const dataset::DatasetBaseList& datasets) {
  uint64_t first_dataset_len = datasets.front()->len();
  for (const auto& dataset : datasets) {
    if (dataset->len() != first_dataset_len) {
      std::stringstream error_msg;
      error_msg << "All passed in datasets must have the same number "
                   "of total examples, but found "
                << dataset->len() << " samples in one dataset and "
                << first_dataset_len << " samples in another.";
      throw std::invalid_argument(error_msg.str());
    }
  }
}

SingleBatchDatasetContext::SingleBatchDatasetContext(
    std::vector<BoltVector>&& data) {
  for (auto vector : data) {
    _data.push_back(BoltBatch({std::move(vector)}));
  }
}

SingleBatchDatasetContext::SingleBatchDatasetContext(
    std::vector<BoltBatch>&& batches)
    : _data(std::move(batches)) {
  uint32_t first_batch_size = _data.front().getBatchSize();
  for (const auto& batch : _data) {
    if (batch.getBatchSize() != first_batch_size) {
      throw std::invalid_argument(
          "All batches must have the same batch size, "
          "but found " +
          std::to_string(first_batch_size) + " for one batch size and " +
          std::to_string(batch.getBatchSize()) + " for another");
    }
  }
}

void SingleBatchDatasetContext::setInputs(uint64_t batch_idx,
                                          const std::vector<InputPtr>& inputs) {
  (void)batch_idx;
  for (uint32_t i = 0; i < inputs.size(); i++) {
    inputs[i]->setInputs(&_data[i]);
  }
}

}  // namespace thirdai::bolt