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

class Metric {
 public:
  virtual void processSample(const BoltVector& output,
                             const BoltVector& labels) = 0;

  virtual double getAndReset(bool verbose) = 0;

  virtual std::string getName() = 0;
};

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

    uint32_t pred =
        output.isDense() ? max_act_index : output.active_neurons[max_act_index];

    if (labels.isDense()) {
      if (labels.activations[pred] > 0) {
        _correct++;
      }
    } else {
      const uint32_t* label_start = labels.active_neurons;
      const uint32_t* label_end = labels.active_neurons + labels.len;
      if (std::find(label_start, label_end, pred) != label_end) {
        _correct++;
      }
    }
    _num_samples++;
  }

  double getAndReset(bool verbose) final {
    double acc = static_cast<double>(_correct) / _num_samples;
    if (verbose) {
      std::cout << "Accuracy: " << acc << " (" << _correct << "/"
                << _num_samples << ")" << std::endl;
    }
    _correct = 0;
    _num_samples = 0;
    return acc;
  }

  static constexpr const char* _name = "categorical_accuracy";

  std::string getName() final { return _name; }

 private:
  std::atomic<uint32_t> _correct;
  std::atomic<uint32_t> _num_samples;
};

class MetricAggregator {
 public:
  explicit MetricAggregator(const std::vector<std::string>& metrics,
                            bool verbose = true)
      : _verbose(verbose) {
    for (const auto& name : metrics) {
      if (name == CategoricalAccuracy::_name) {
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
      _output[m->getName()].push_back(m->getAndReset(_verbose));
    }
  }

  std::unordered_map<std::string, std::vector<double>> getOutput() {
    return _output;
  }

 private:
  std::vector<std::unique_ptr<Metric>> _metrics;
  std::unordered_map<std::string, std::vector<double>> _output;
  bool _verbose;
};

}  // namespace thirdai::bolt