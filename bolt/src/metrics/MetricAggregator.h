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
<<<<<<< HEAD
      if (name == CategoricalAccuracy::name) {
        _metrics.push_back(std::make_shared<CategoricalAccuracy>());
      } else if (name == WeightedMeanAbsolutePercentageError::name) {
        _metrics.push_back(
            std::make_shared<WeightedMeanAbsolutePercentageError>());
      } else if (name == MeanSquaredErrorMetric::name) {
        _metrics.push_back(std::make_shared<MeanSquaredErrorMetric>());
      } else if (FMeasure::isFMeasure(name)) {
        _metrics.push_back(FMeasure::make(name));
      } else if (RecallAtK::isRecallAtK(name)) {
        _metrics.push_back(RecallAtK::make(name));
      } else if (SampledRecallAtK::isSampledRecallAtK(name)) {
        _metrics.push_back(SampledRecallAtK::make(name));
      } else {
        throw std::invalid_argument("'" + name + "' is not a valid metric.");
      }
=======
      _metrics.push_back(makeMetric(name));
>>>>>>> 56f2b447317f6447c102498eb69c1187140b7e50
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

 private:
  std::vector<std::shared_ptr<Metric>> _metrics;
  MetricData _output;
};

}  // namespace thirdai::bolt
