#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt {

class Input;
using InputPtr = std::shared_ptr<Input>;

/**
 * This class provides an interface to interact with dataset objects,
 * particularly as a common interface for the SingleUnitDatasetContext and the
 * DatasetContext objects
 */
class DatasetContextBase {
 public:
  virtual void setInputs(uint64_t batch_idx,
                         const std::vector<InputPtr>& inputs) = 0;

  virtual uint64_t numVectorDatasets() const = 0;

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
                 dataset::BoltDatasetPtr labels);

  void setInputs(uint64_t batch_idx,
                 const std::vector<InputPtr>& inputs) override;

  uint64_t batchSize() const { return _all_dag_datasets.front()->batchSize(); }

  uint64_t batchSize(uint64_t batch_idx) const {
    return _all_dag_datasets.front()->batchSize(batch_idx);
  }

  uint64_t len() const { return _all_dag_datasets.front()->len(); }

  uint64_t numBatches() const {
    return _all_dag_datasets.front()->numBatches();
  }

  uint64_t numVectorDatasets() const override { return _data.size(); }

  const dataset::BoltDatasetPtr& labels() const { return _labels; }

 private:
  static void verifyBatchSizes(const dataset::BoltDatasetList& datasets);
  static void verifyDatasetLens(const dataset::BoltDatasetList& datasets);

  std::vector<dataset::BoltDatasetPtr> _data;
  dataset::BoltDatasetPtr _labels;
  std::vector<dataset::BoltDatasetPtr> _all_dag_datasets;
};

/**
 * This class provides the interface from DatasetContextBase but is constructed
 * assuming a single sample input for inference.
 */
class SingleBatchDatasetContext final : public DatasetContextBase {
 public:
  explicit SingleBatchDatasetContext(std::vector<BoltVector>&& data);

  explicit SingleBatchDatasetContext(std::vector<BoltBatch>&& batches);

  void setInputs(uint64_t batch_idx,
                 const std::vector<InputPtr>& inputs) override;

  uint64_t numVectorDatasets() const override { return _data.size(); }

  uint32_t batchSize() const { return _data.front().getBatchSize(); }

  std::vector<BoltBatch> _data;
};

}  // namespace thirdai::bolt
