#include "Metric.h"
#include <bolt/src/train/metrics/CategoricalAccuracy.h>
#include <stdexcept>

namespace thirdai::bolt::train::metrics {

void Metric::incrementAtomicFloat(std::atomic<float>& value, float increment) {
  float curr_val = value.load(std::memory_order_relaxed);
  // Compare and exchange weak will update the value of curr_val if it is found
  // to be different from the value stored in the atomic variable.
  while (!value.compare_exchange_weak(curr_val, curr_val + increment,
                                      std::memory_order_relaxed)) {
  }
}

MetricCollection::MetricCollection(const InputMetrics& metrics) {
  for (const auto& [name, metric] : metrics) {
    metric->setName(name);
    _metrics.push_back(metric);
  }
}

void MetricCollection::recordBatch(uint32_t batch_size) {
#pragma omp parallel for default(none) shared(batch_size)
  for (uint32_t i = 0; i < batch_size; i++) {
    for (const auto& metric : _metrics) {
      metric->record(i);
    }
  }
}

void MetricCollection::updateHistory(History& history) {
  for (const auto& metric : _metrics) {
    history[metric->name()].push_back(metric->value());
  }
}

std::string MetricCollection::summarizeLastStep() const {
  std::stringstream summary;

  for (const auto& metric : _metrics) {
    summary << metric->name() << "=" << metric->value() << " ";
  }

  return summary.str();
}

void MetricCollection::reset() {
  for (auto& metric : _metrics) {
    metric->reset();
  }
}

InputMetrics metricsForSingleOutputModel(
    const std::vector<std::string>& metric_names,
    const nn::autograd::ComputationPtr& output,
    const nn::autograd::ComputationPtr& labels) {
  InputMetrics metrics;

  for (const auto& name : metric_names) {
    if (name == "categorical_accuracy") {
      metrics[name] = std::make_shared<CategoricalAccuracy>(output, labels);
    } else {
      throw std::invalid_argument("Metric '" + name +
                                  "' is not yet supported.");
    }
  }

  return metrics;
}

float divideTwoAtomicIntegers(const std::atomic_uint32_t& numerator,
                              const std::atomic_uint32_t& denominator) {
  uint32_t loaded_numerator = numerator.load();
  uint32_t loaded_denominator = denominator.load();

  if (loaded_denominator == 0) {
    return 0.0;
  }

  return static_cast<float>(loaded_numerator) / loaded_denominator;
}

uint32_t truePositivesInTopK(TopKActivationsQueue& top_k_predictions,
                             const BoltVector& label) {
  uint32_t true_positives = 0;
  while (!top_k_predictions.empty()) {
    ValueIndexPair valueIndex = top_k_predictions.top();
    uint32_t prediction = valueIndex.second;
    if (label.findActiveNeuronNoTemplate(prediction).activation > 0) {
      true_positives++;
    }
    top_k_predictions.pop();
  }
  return true_positives;
}

}  // namespace thirdai::bolt::train::metrics