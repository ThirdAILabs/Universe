#pragma once

#include "Metric.h"
#include <bolt/src/loss_functions/LossFunctions.h>
#include <memory>
#include <regex>

namespace thirdai::bolt {

using MetricData = std::unordered_map<std::string, std::vector<double>>;
using InferenceMetricData = std::unordered_map<std::string, double>;

// TODO(Geordie): Instead of hard coding the options, use a static map.
class MetricAggregator {
 public:
  // Loss function metrics are only supported during training. See comments in
  // loss function for why. A nullptr is passed in during testing to indicate
  // that it is not avilable.
  explicit MetricAggregator(const std::vector<std::string>& metrics) {
    for (const auto& name : metrics) {
      _metrics.push_back(makeMetric(name));
    }
  }

  void processSample(const BoltVector& output, const BoltVector& labels) {
    for (auto& metric : _metrics) {
      metric->record(output, labels);
    }
  }

  void logAndReset() {
    for (auto& metric : _metrics) {
      _output[metric->name()].push_back(metric->value());
      metric->reset();
    }
  }

  void logBatchMetrics() {
    for (auto& metric : _metrics) {
      _batch_output[metric->name()].push_back(metric->value());
    }
  }

  std::string summary() {
    std::stringstream stream;
    stream << "{";
    for (size_t i = 0; i < _metrics.size(); i++) {
      if (i != 0) {
        stream << ", ";
      }
      stream << _metrics[i]->summary();
    }
    stream << "}";
    return stream.str();
  }

  MetricData getOutput() { return _output; }

  MetricData getBatchOutput() { return _batch_output; }

  std::vector<double>& getSingleOutput(const std::string& metric_name) {
    if (_output.count(metric_name) != 0) {
      return _output[metric_name];
    }
    throw std::invalid_argument("Could not find metric name '" + metric_name +
                                "' in list of computed metrics.");
  }

  InferenceMetricData getOutputFromInference() {
    InferenceMetricData data;
    for (const auto& metric : _output) {
      data[metric.first] = metric.second.at(0);
    }
    return data;
  }

  uint32_t getNumMetricsTracked() { return _metrics.size(); }

  std::vector<std::shared_ptr<Metric>> getMetrics() const { return _metrics; }

 private:
  std::vector<std::shared_ptr<Metric>> _metrics;
  MetricData _output;
  MetricData _batch_output;
};

}  // namespace thirdai::bolt
