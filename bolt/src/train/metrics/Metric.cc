#include "Metric.h"
#include <stdexcept>

namespace thirdai::bolt::train::metrics {

void Metric::record(uint32_t index_in_batch) {
  record(_outputs->getVector(index_in_batch),
         _labels->getVector(index_in_batch));
}

void Metric::setOutputs(nn::tensor::ActivationTensorPtr outputs) {
  if (_outputs != outputs) {
    throw std::runtime_error(
        "Cannot rebind the metric to a new model or a new output in the same "
        "model");
  }
  _outputs = std::move(outputs);
}

void Metric::setLabels(nn::tensor::InputTensorPtr labels) {
  if (_labels != labels) {
    throw std::runtime_error(
        "Cannot rebind the metric to a new model or a new output in the same "
        "model");
  }
  _labels = std::move(labels);
}

MetricList::MetricList(
    const std::unordered_map<std::string, std::vector<MetricPtr>>& metrics,
    const nn::model::ModelPtr& model) {
  for (const auto& output_metrics : metrics) {
    for (const auto& metric : output_metrics.second) {
      metric->setOutputs(model->getTensor(output_metrics.first));
      metric->setLabels(model->getLabelsForOutput(output_metrics.first));
      _metrics.push_back(metric);
    }
  }
}

void MetricList::recordLastBatch(uint32_t batch_size) {
#pragma omp parallel for default(none) shared(batch_size)
  for (uint32_t i = 0; i < batch_size; i++) {
    for (const auto& metric : _metrics) {
      metric->record(i);
    }
  }
}

}  // namespace thirdai::bolt::train::metrics