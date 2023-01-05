#include "Metric.h"
#include <atomic>
#include <stdexcept>

namespace thirdai::bolt::train::metrics {

void Metric::incrementAtomicFloat(std::atomic<float>& value, float increment) {
  bool success;
  do {
    float curr_val = value.load(std::memory_order_relaxed);
    success = value.compare_exchange_weak(curr_val, curr_val + increment,
                                          std::memory_order_relaxed);
  } while (!success);
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

void MetricList::recordBatch(uint32_t batch_size) {
#pragma omp parallel for default(none) shared(batch_size)
  for (uint32_t i = 0; i < batch_size; i++) {
    for (const auto& metric : _metrics) {
      metric->record(i);
    }
  }
}

void MetricList::updateHistory(std::shared_ptr<History>& history,
                               const std::string& prefix) {
  for (const auto& metric : _metrics) {
    (*history)[metric->name()][prefix + metric->outputName()].push_back(
        metric->value());
  }
}

std::string MetricList::summarizeLastStep() const {
  std::stringstream summary;

  for (const auto& metric : _metrics) {
    summary << metric->outputName() << "::" << metric->name() << "=";
    summary << metric->value() << " ";
  }

  return summary.str();
}

void MetricList::reset() {
  for (auto& metric : _metrics) {
    metric->reset();
  }
}

}  // namespace thirdai::bolt::train::metrics