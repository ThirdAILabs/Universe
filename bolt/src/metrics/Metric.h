#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

// Metric interface
class Metric {
 public:
  // Computes and updates the value of the metric given the sample.
  // For instance this may update the accuracy.
  virtual void processSample(const BoltVector& output,
                             const BoltVector& labels) = 0;

  // Returns the value of the metric and resets it. For instance this would be
  // called ad the end of each epoch.
  virtual double getMetricAndReset(bool verbose) = 0;

  // Returns the name of the metric.
  virtual std::string getName() = 0;

  virtual ~Metric() = default;
};

/**
 * The categorical accuracy is the accuracy @1 which measures for what fraction
 * of the samples the neuron with the highest activation is in the labels.
 */
class CategoricalAccuracy final : public Metric {
 public:
  CategoricalAccuracy() : _correct(0), _num_samples(0) {}

  void processSample(const BoltVector& output, const BoltVector& labels) final {
    float max_act = std::numeric_limits<float>::min();
    uint32_t max_act_index = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < output.len; i++) {
      if (output.activations[i] > max_act) {
        max_act = output.activations[i];
        max_act_index = i;
      }
    }

    // The nueron with the largest activation is the prediction
    uint32_t pred =
        output.isDense() ? max_act_index : output.active_neurons[max_act_index];

    if (labels.isDense()) {
      // If labels are dense we check if the predection has a non-zero label.
      if (labels.activations[pred] > 0) {
        _correct++;
      }
    } else {
      // If the labels are sparse then we have to search the list of labels for
      // the prediction.
      const uint32_t* label_start = labels.active_neurons;
      const uint32_t* label_end = labels.active_neurons + labels.len;
      if (std::find(label_start, label_end, pred) != label_end) {
        _correct++;
      }
    }
    _num_samples++;
  }

  double getMetricAndReset(bool verbose) final {
    double acc = static_cast<double>(_correct) / _num_samples;
    if (verbose) {
      std::cout << "Accuracy: " << acc << " (" << _correct << "/"
                << _num_samples << ")" << std::endl;
    }
    _correct = 0;
    _num_samples = 0;
    return acc;
  }

  static constexpr const char* name = "categorical_accuracy";

  std::string getName() final { return name; }

 private:
  std::atomic<uint32_t> _correct;
  std::atomic<uint32_t> _num_samples;
};

/**
 * The weighted mean absolute percentage error is a regression error that measures
 * the absolute deviation of the prediction from the ground truth
 */
class WeightedMeanAbsolutePercentageError final : public Metric {
 public:
  CategoricalAccuracy() : _correct(0), _num_samples(0) {}

  void processSample(const BoltVector& output, const BoltVector& labels) final {
    float max_act = std::numeric_limits<float>::min();
    uint32_t max_act_index = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < output.len; i++) {
      if (output.activations[i] > max_act) {
        max_act = output.activations[i];
        max_act_index = i;
      }
    }

    // The nueron with the largest activation is the prediction
    uint32_t pred =
        output.isDense() ? max_act_index : output.active_neurons[max_act_index];

    if (labels.isDense()) {
      // If labels are dense we check if the predection has a non-zero label.
      if (labels.activations[pred] > 0) {
        _correct++;
      }
    } else {
      // If the labels are sparse then we have to search the list of labels for
      // the prediction.
      const uint32_t* label_start = labels.active_neurons;
      const uint32_t* label_end = labels.active_neurons + labels.len;
      if (std::find(label_start, label_end, pred) != label_end) {
        _correct++;
      }
    }
    _num_samples++;
  }

  double getMetricAndReset(bool verbose) final {
    double acc = static_cast<double>(_correct) / _num_samples;
    if (verbose) {
      std::cout << "Accuracy: " << acc << " (" << _correct << "/"
                << _num_samples << ")" << std::endl;
    }
    _correct = 0;
    _num_samples = 0;
    return acc;
  }

  static constexpr const char* name = "categorical_accuracy";

  std::string getName() final { return name; }

 private:
  std::atomic<uint32_t> _correct;
  std::atomic<uint32_t> _num_samples;
};


using MetricData = std::unordered_map<std::string, std::vector<double>>;

class MetricAggregator {
 public:
  explicit MetricAggregator(const std::vector<std::string>& metrics,
                            bool verbose = true)
      : _verbose(verbose) {
    for (const auto& name : metrics) {
      if (name == CategoricalAccuracy::name) {
        _metrics.push_back(std::make_unique<CategoricalAccuracy>());
      } else {
        throw std::invalid_argument("'" + name + "' is not a valid metric.");
      }
    }
  }

  void processSample(const BoltVector& output, const BoltVector& labels) {
    for (auto& m : _metrics) {
      m->processSample(output, labels);
    }
  }

  void logAndReset() {
    for (auto& m : _metrics) {
      _output[m->getName()].push_back(m->getMetricAndReset(_verbose));
    }
  }

  MetricData getOutput() { return _output; }

 private:
  std::vector<std::unique_ptr<Metric>> _metrics;
  MetricData _output;
  bool _verbose;
};

}  // namespace thirdai::bolt