#pragma once

#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <dataset/src/Datasets.h>
#include <exceptions/src/Exceptions.h>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::bolt::train {

using LabeldBoltDataset =
    std::pair<dataset::BoltDatasetList, dataset::BoltDatasetPtr>;

class DistributedTrainingWrapper {
 public:
  DistributedTrainingWrapper(const nn::model::ModelPtr& model,
                             const TrainConfig& train_config,
                             uint32_t worker_id);

  void computeAndStoreBatchGradients(uint32_t batch_idx);

  void updateParameters();

  std::unordered_map<std::string, float> validationAndSaveBest();

  nn::model::ModelPtr getModel() { return _model; }

  void finishTraining() const {}

  uint64_t numBatches();

  void setDatasets(const dataset::BoltDatasetList& train_data,
                   const dataset::BoltDatasetPtr& train_labels) {
    _train_data = convertLabeldData(train_data, train_labels);
  }

  void freezeHashTables(bool insert_labels_if_not_found) {  // NOLINT
    (void)insert_labels_if_not_found;
    throw exceptions::NotImplemented(
        "Bolt V2 Distributed Trainer: freezeHashTables.");
  }

  std::unordered_map<std::string, std::vector<float>> getTrainingMetrics() {
    _train_metrics.updateHistory(_train_metric_history);
    _train_metrics.reset();

    return _train_metric_history;
  }

  std::pair<const float*, uint64_t> getGradients() const;

  void setGradients(const float* new_grad, uint64_t flattened_dim);

 private:
  std::optional<LabeledDataset> convertLabeldData(
      const dataset::BoltDatasetList& data,
      const dataset::BoltDatasetPtr& labels);

  static uint64_t sumFlattenedDims(
      const std::vector<std::vector<float>*>& grads);

  bool shouldLogMetrics() const {
    return _worker_id == 0 && _logging_interval &&
           ((_model->trainSteps() % *_logging_interval) ==
            (*_logging_interval - 1));
  }

  nn::model::ModelPtr _model;
  uint32_t _worker_id;

  float _learning_rate;
  metrics::MetricCollection _train_metrics;
  std::optional<uint32_t> _logging_interval;
  std::vector<std::string> _validation_metrics;
  bool _use_sparsity_in_validation;

  metrics::History _train_metric_history;

  std::optional<LabeledDataset> _train_data;
  std::optional<LabeledDataset> _validation_data;
};

using DistributedTrainingWrapperPtr =
    std::shared_ptr<DistributedTrainingWrapper>;

}  // namespace thirdai::bolt::train