#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <dataset/src/Datasets.h>
#include <exceptions/src/Exceptions.h>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::bolt::train {

using LabeldBoltDataset =
    std::pair<dataset::BoltDatasetList, dataset::BoltDatasetPtr>;

class DistributedTrainingWrapper {
 public:
  DistributedTrainingWrapper(
      nn::model::ModelPtr model, float learning_rate,
      const metrics::InputMetrics& input_metrics = {},
      const std::optional<LabeldBoltDataset>& validation_data = std::nullopt,
      metrics::InputMetrics validation_metrics = {},
      bool use_sparsity_in_validation = false)
      : _model(std::move(model)),
        _learning_rate(learning_rate),
        _train_metrics(input_metrics),
        _validation_metrics(std::move(validation_metrics)),
        _use_sparsity_in_validation(use_sparsity_in_validation) {
    if (_model->outputs().size() != 1) {
      throw std::invalid_argument(
          "Distributed training is currently only supported for models with a "
          "single output.");
    }

    if (validation_data) {
      _validation_data =
          convertLabeldData(validation_data->first, validation_data->second);
    }
  }

  void computeAndStoreBatchGradients(uint32_t batch_idx) {
    if (numBatches() <= batch_idx) {
      throw std::invalid_argument(
          "Cannot compute gradients for invalid batch index: " +
          std::to_string(batch_idx) + ".");
    }

    const nn::tensor::TensorList& inputs = _train_data->first.at(batch_idx);
    const nn::tensor::TensorList& labels = _train_data->second.at(batch_idx);

    _model->trainOnBatch(inputs, labels);

    _train_metrics.recordBatch(inputs.at(0)->batchSize());
  }

  void updateParameters() { _model->updateParameters(_learning_rate); }

  std::unordered_map<std::string, float> validationAndSaveBest() {
    if (!_validation_data) {
      return {};
    }

    Trainer trainer(_model);
    auto history = trainer.validate(*_validation_data, _validation_metrics,
                                    _use_sparsity_in_validation);

    std::unordered_map<std::string, float> last_metrics;
    for (const auto& [metric_name, metric_vals] : history) {
      last_metrics[metric_name] = metric_vals.back();
    }
    return last_metrics;
  }

  nn::model::ModelPtr getModel() { return _model; }

  uint64_t numBatches() {
    if (!_train_data) {
      return 0;
    }
    return _train_data->first.size();
  }

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

 private:
  std::optional<LabeledDataset> convertLabeldData(
      const dataset::BoltDatasetList& data,
      const dataset::BoltDatasetPtr& labels) {
    auto data_tensors = convertDatasets(data, _model->inputDims());
    auto label_tensors = convertDataset(labels, _model->outputs().at(0)->dim());

    return std::make_optional<LabeledDataset>(std::move(data_tensors),
                                              std::move(label_tensors));
  }

  nn::model::ModelPtr _model;

  float _learning_rate;
  metrics::MetricCollection _train_metrics;
  metrics::InputMetrics _validation_metrics;
  bool _use_sparsity_in_validation;

  metrics::History _train_metric_history;

  std::optional<LabeledDataset> _train_data;
  std::optional<LabeledDataset> _validation_data;
};

}  // namespace thirdai::bolt::train