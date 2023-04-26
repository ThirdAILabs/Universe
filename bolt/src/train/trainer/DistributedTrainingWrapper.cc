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
      _steps_since_save(0),
      _save_context(train_config.saveContext()) {
  if (_model->outputs().size() != 1) {
    throw std::invalid_argument(
        "Distributed training is currently only supported for models with a "
        "single output.");
  }

  for (const auto& op : model->ops()) {
    op->disableSparseParameterUpdates();
  }

  if (auto validation = train_config.getValidationContext()) {
    _validation_data =
        convertLabeldData(validation->data(), validation->labels());
    _validation_metrics = validation->config().getMetricNames();
    _use_sparsity_in_validation =
        validation->config().shouldReturnActivations();
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

  ++_steps_since_save;
}

std::unordered_map<std::string, float>
DistributedTrainingWrapper::validationAndSave() {
  if (_save_context && (_steps_since_save % _save_context->frequency()) == 0) {
    std::string checkpoint_path =
        _save_context->prefix() + "_" + std::to_string(_steps_since_save) + ".checkpoint.bolt";
    std::string save_path =
        _save_context->prefix() + "_" + std::to_string(_steps_since_save) + ".save.bolt";
    _model->checkpoint(checkpoint_path);
    _model->save(save_path);
  }

  if (!_validation_data) {
    return {};
  }

  Trainer trainer(_model);
  auto history = trainer.validate_with_metric_names(
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
  auto grads = _model->gradients();

  uint64_t total_dim = sumFlattenedDims(grads);

  float* combined_grads = new float[total_dim];
  uint64_t offset = 0;
  for (const auto* grad : grads) {
    std::copy(grad->data(), grad->data() + grad->size(),
              combined_grads + offset);
    offset += grad->size();
  }

  return {combined_grads, total_dim};
}

void DistributedTrainingWrapper::setGradients(const float* new_grad,
                                              uint64_t flattened_dim) {
  auto grads = _model->gradients();

  uint64_t total_dim = sumFlattenedDims(grads);

  if (total_dim != flattened_dim) {
    std::stringstream error;
    error << "Expected " << total_dim
          << " parameters in setGradients, but received " << flattened_dim
          << " parameters.";
    throw std::invalid_argument(error.str());
  }

  uint64_t offset = 0;
  for (auto* grad : grads) {
    std::copy(new_grad + offset, new_grad + offset + grad->size(),
              grad->data());
    offset += grad->size();
  }
}

std::optional<LabeledDataset> DistributedTrainingWrapper::convertLabeldData(
    const dataset::BoltDatasetList& data,
    const dataset::BoltDatasetPtr& labels) {
  auto data_tensors = convertDatasets(data, _model->inputDims());
  auto label_tensors = convertDataset(labels, _model->outputs().at(0)->dim());

  return std::make_optional<LabeledDataset>(std::move(data_tensors),
                                            std::move(label_tensors));
}

uint64_t DistributedTrainingWrapper::sumFlattenedDims(
    const std::vector<std::vector<float>*>& grads) {
  uint64_t total_dim = 0;
  for (const auto* grad : grads) {
    total_dim += grad->size();
  }
  return total_dim;
}

}  // namespace thirdai::bolt::train
