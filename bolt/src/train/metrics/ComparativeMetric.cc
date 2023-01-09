#include "ComparativeMetric.h"

namespace thirdai::bolt::train::metrics {

void ComparativeMetric::record(uint32_t index_in_batch) {
  record(_outputs->getVector(index_in_batch),
         _labels->getVector(index_in_batch));
}

void ComparativeMetric::setOutputs(nn::tensor::ActivationTensorPtr outputs) {
  if (_outputs && _outputs != outputs) {
    throw std::runtime_error(
        "Cannot rebind the metric to a new model or a new output in the same "
        "model");
  }
  _outputs = std::move(outputs);
}

void ComparativeMetric::setLabels(nn::tensor::InputTensorPtr labels) {
  if (_labels && _labels != labels) {
    throw std::runtime_error(
        "Cannot rebind the metric to a new model or a new output in the same "
        "model");
  }
  _labels = std::move(labels);
}

std::string ComparativeMetric::outputName() const { return _outputs->name(); }

}  // namespace thirdai::bolt::train::metrics