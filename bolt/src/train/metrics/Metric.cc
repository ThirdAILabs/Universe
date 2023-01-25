#include "Metric.h"
#include <atomic>
#include <stdexcept>

namespace thirdai::bolt::train::metrics {

void Metric::incrementAtomicFloat(std::atomic<float>& value, float increment) {
  float curr_val = value.load(std::memory_order_relaxed);
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

void MetricCollection::updateHistory(std::shared_ptr<History>& history,
                                     const std::string& prefix) {
  for (const auto& metric : _metrics) {
    (*history)[prefix + metric->name()].push_back(metric->value());
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

}  // namespace thirdai::bolt::train::metrics