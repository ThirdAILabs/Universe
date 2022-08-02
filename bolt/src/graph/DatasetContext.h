#pragma once

#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/nodes/TokenInput.h>
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt {

/**
 * This class provides an interface to interact with dataset objects,
 * particularly as a common interface for the SingleUnitDatasetContext and the
 * DatasetContext objects
 */
class DatasetContextBase {
 public:
  virtual void setInputs(uint64_t batch_idx,
                         const std::vector<InputPtr>& inputs,
                         const std::vector<TokenInputPtr>& token_inputs) = 0;

  virtual uint64_t numVectorDatasets() const = 0;

  virtual uint64_t numTokenDatasets() const = 0;

  virtual ~DatasetContextBase() = default;
};

/**
 * This class stores information about the datasets passed into train or
 * predict. This is to provide a simple interface to interact with the
 * datasets, and also performs checks that they all have the same length,
 * batch size, etc. Finally it provides methods for obtaining the length and
 * batch size of the datasets once they are verified to be correct.
 */
class DatasetContext final : public DatasetContextBase {
 public:
  DatasetContext(std::vector<dataset::BoltDatasetPtr> data,
                 std::vector<dataset::BoltTokenDatasetPtr> tokens,
                 dataset::BoltDatasetPtr labels)
      : _data(std::move(data)),
        _tokens(std::move(tokens)),
        _labels(std::move(labels)) {
    _all_dag_datasets.insert(_all_dag_datasets.end(), _data.begin(),
                             _data.end());
    _all_dag_datasets.insert(_all_dag_datasets.end(), _tokens.begin(),
                             _tokens.end());
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

  void setInputs(uint64_t batch_idx, const std::vector<InputPtr>& inputs,
                 const std::vector<TokenInputPtr>& token_inputs) override {
    for (uint32_t i = 0; i < inputs.size(); i++) {
      inputs[i]->setInputs(&_data[i]->at(batch_idx));
    }
    for (uint32_t i = 0; i < token_inputs.size(); i++) {
      token_inputs[i]->setTokenInputs(&_tokens[i]->at(batch_idx));
    }
  }

  virtual uint64_t batchSize() const {
    return _all_dag_datasets.front()->batchSize();
  }

  virtual uint64_t batchSize(uint64_t batch_idx) const {
    return _all_dag_datasets.front()->batchSize(batch_idx);
  }

  virtual uint64_t len() const { return _all_dag_datasets.front()->len(); }

  virtual uint64_t numBatches() const {
    return _all_dag_datasets.front()->numBatches();
  }

  uint64_t numVectorDatasets() const override { return _data.size(); }

  uint64_t numTokenDatasets() const override { return _tokens.size(); }

  const dataset::BoltDatasetPtr& labels() const { return _labels; }

 private:
  static void verifyBatchSizes(const dataset::DatasetBaseList& datasets) {
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

  static void verifyDatasetLens(const dataset::DatasetBaseList& datasets) {
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

  std::vector<dataset::BoltDatasetPtr> _data;
  std::vector<dataset::BoltTokenDatasetPtr> _tokens;
  dataset::BoltDatasetPtr _labels;
  std::vector<dataset::DatasetBasePtr> _all_dag_datasets;
};

/**
 * This class provides the interface from DatasetContextBase but is constructed
 * assuming a single sample input for inference.
 */
class SingleUnitDatasetContext final : public DatasetContextBase {
 public:
  SingleUnitDatasetContext(std::vector<BoltVector>&& data,
                           std::vector<std::vector<uint32_t>>&& tokens) {
    for (auto vector : data) {
      _data.push_back(BoltBatch({std::move(vector)}));
    }

    for (auto vector : tokens) {
      _tokens.push_back(dataset::BoltTokenBatch({std::move(vector)}));
    }

    if (_data.empty() && _tokens.empty()) {
      throw std::invalid_argument(
          "Must pass in at least one dataset, but found 0.");
    }
  }

  void setInputs(uint64_t batch_idx, const std::vector<InputPtr>& inputs,
                 const std::vector<TokenInputPtr>& token_inputs) override {
    (void)batch_idx;
    for (uint32_t i = 0; i < inputs.size(); i++) {
      inputs[i]->setInputs(&_data[i]);
    }
    for (uint32_t i = 0; i < token_inputs.size(); i++) {
      token_inputs[i]->setTokenInputs(&_tokens[i]);
    }
  }

  uint64_t numVectorDatasets() const override { return _data.size(); }

  uint64_t numTokenDatasets() const override { return _tokens.size(); }

  std::vector<BoltBatch> _data;
  std::vector<dataset::BoltTokenBatch> _tokens;
};

}  // namespace thirdai::bolt