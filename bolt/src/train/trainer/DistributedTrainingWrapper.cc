#include "DistributedTrainingWrapper.h"
#include <bolt/src/train/metrics/Metric.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/CheckLicense.h>
#include <utils/Logging.h>

namespace thirdai::bolt::train {

DistributedTrainingWrapper::DistributedTrainingWrapper(
    const nn::model::ModelPtr& model, const TrainConfig& train_config,
    uint32_t worker_id)
    : _model(model),
      _worker_id(worker_id),
      _learning_rate(train_config.learningRate()),
      _train_metrics(metrics::fromMetricNames(model, train_config.metrics())),
      _logging_interval(train_config.logLossFrequency()),
      _use_sparsity_in_validation(false),
      _trainer(model) {
  if (_model->outputs().size() != 1) {
    throw std::invalid_argument(
        "Distributed training is currently only supported for models with a "
        "single output.");
  }

  model->disableSparseParameterUpdates();

  if (auto validation = train_config.getValidationContext()) {
    _validation_data =
        convertLabeldData(validation->data(), validation->labels());
    _validation_metrics = validation->config().getMetricNames();
    _use_sparsity_in_validation =
        validation->config().shouldReturnActivations();
  }

  // TODO(Nicholas): add saving and best metric tracking.
  if (train_config.saveContext()) {
    throw exceptions::NotImplemented(
        "Training with a save context is not yet supported for bolt v2 "
        "distributed.");
  }
}

void DistributedTrainingWrapper::computeAndStoreBatchGradients(
    uint32_t batch_idx) {
  licensing::entitlements().verifyFullAccess();

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

void DistributedTrainingWrapper::updateParameters() {
  _model->updateParameters(_learning_rate);

  if (shouldLogMetrics()) {
    logging::info("train | train_steps {} | {}", _model->trainSteps(),
                  _train_metrics.summarizeLastStep());
  }
}

std::unordered_map<std::string, float>
DistributedTrainingWrapper::validationAndSaveBest() {
  if (!_validation_data) {
    return {};
  }

  auto history = _trainer.validate_with_metric_names(
      *_validation_data, _validation_metrics, _use_sparsity_in_validation);

  std::unordered_map<std::string, float> last_metrics;
  for (const auto& [metric_name, metric_vals] : history) {
    last_metrics[metric_name] = metric_vals.back();
  }

  return last_metrics;
}

uint64_t DistributedTrainingWrapper::numBatches() {
  if (!_train_data) {
    return 0;
  }
  return _train_data->first.size();
}

std::pair<const float*, uint64_t> DistributedTrainingWrapper::getGradients()
    const {
  return _model->getFlattenedGradients();
}

void DistributedTrainingWrapper::setGradients(const float* new_grad,
                                              uint64_t flattened_dim) {
  _model->setFlattenedGradients(new_grad, flattened_dim);
}

std::optional<LabeledDataset> DistributedTrainingWrapper::convertLabeldData(
    const dataset::BoltDatasetList& data,
    const dataset::BoltDatasetPtr& labels) {
  auto data_tensors = convertDatasets(data, _model->inputDims());
  auto label_tensors = convertDataset(labels, _model->outputs().at(0)->dim());

  return std::make_optional<LabeledDataset>(std::move(data_tensors),
                                            std::move(label_tensors));
}

}  // namespace thirdai::bolt::train
