#pragma once

#include "Metric.h"
#include <bolt/src/loss_functions/LossFunctions.h>

namespace thirdai::bolt {

using MetricData = std::unordered_map<std::string, std::vector<double>>;
using InferenceMetricData = std::unordered_map<std::string, double>;

// TODO(Geordie): Instead of hard coding the options, use a static map.
class MetricAggregator {
 public:
  // Loss function metrics are only supported during training. See comments in
  // loss function for why. A nullptr is passed in during testing to indicate
  // that it is not avilable.
  explicit MetricAggregator(const std::vector<std::string>& metrics,
                            bool verbose = true)
      : _verbose(verbose) {
    for (const auto& name : metrics) {
      if (name == CategoricalAccuracy::name) {
        _metrics.push_back(std::make_shared<CategoricalAccuracy>());
      } else if (name == WeightedMeanAbsolutePercentageError::name) {
        _metrics.push_back(
            std::make_shared<WeightedMeanAbsolutePercentageError>());
      } else if (name == MeanSquaredErrorMetric::name) {
        _metrics.push_back(std::make_shared<MeanSquaredErrorMetric>());
      } else if (RecallAt::isRecallAtK(name)) {
        _metrics.push_back(std::make_shared<RecallAt>(RecallAt::getK(name)));
      } else {
        throw std::invalid_argument("'" + name + "' is not a valid metric.");
      }
    }
  }

  void processSample(const BoltVector& output, const BoltVector& labels) {
    for (auto& m : _metrics) {
      m->computeMetric(output, labels);
    }
  }

  void logAndReset() {
    for (auto& m : _metrics) {
      _output[m->getName()].push_back(m->getMetricAndReset(_verbose));
    }
  }

  MetricData getOutput() { return _output; }

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
  bool _verbose;
};

}  // namespace thirdai::bolt