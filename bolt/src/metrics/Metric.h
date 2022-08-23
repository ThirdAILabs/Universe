#pragma once
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/metrics/MetricHelpers.h>
#include <sys/types.h>
#include <algorithm>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::bolt {

// Metric interface
class Metric {
 public:
  // Computes and updates the value of the metric given the sample.
  // For instance this may update the accuracy.
  virtual void computeMetric(const BoltVector& output,
                             const BoltVector& labels) = 0;

  // Returns the value of the metric and resets it. For instance this would be
  // called at the end of each epoch.
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

  void computeMetric(const BoltVector& output, const BoltVector& labels) final {
    float max_act = -std::numeric_limits<float>::max();
    std::optional<uint32_t> max_act_index = std::nullopt;
    for (uint32_t i = 0; i < output.len; i++) {
      if (output.activations[i] > max_act) {
        max_act = output.activations[i];
        max_act_index = i;
      }
    }

    if (!max_act_index) {
      throw std::runtime_error(
          "Unable to find a output activation larger than the minimum "
          "representable float. This is likely due to a Nan or incorrect "
          "activation function in the final layer.");
    }

    // The nueron with the largest activation is the prediction
    uint32_t pred = output.isDense() ? *max_act_index
                                     : output.active_neurons[*max_act_index];

    if (labels.isDense()) {
      // If labels are dense we check if the prediction has a non-zero label.
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

class MeanSquaredErrorMetric final : public Metric {
 public:
  MeanSquaredErrorMetric() : _mse(0), _num_samples(0) {}

  void computeMetric(const BoltVector& output, const BoltVector& labels) final {
    float error;
    if (output.isDense()) {
      if (labels.isDense()) {
        error = computeMSE<true, true>(output, labels);
      } else {
        error = computeMSE<true, false>(output, labels);
      }
    } else {
      if (labels.isDense()) {
        error = computeMSE<false, true>(output, labels);

      } else {
        error = computeMSE<false, false>(output, labels);
      }
    }

    MetricUtilities::incrementAtomicFloat(_mse, error);
    _num_samples++;
  }

  double getMetricAndReset(bool verbose) final {
    double error = _mse / _num_samples;
    if (verbose) {
      std::cout << "MSE: " << error << std::endl;
    }
    _mse = 0;
    _num_samples = 0;
    return error;
  }

  static constexpr const char* name = "mean_squared_error";

  std::string getName() final { return name; }

 private:
  template <bool DENSE, bool LABEL_DENSE>
  float computeMSE(const BoltVector& output, const BoltVector& labels) {
    if (DENSE || LABEL_DENSE) {
      // If either vector is dense then we need to iterate over the full
      // dimension from the layer.
      uint32_t dim = std::max(output.len, labels.len);

      float error = 0.0;
      for (uint32_t i = 0; i < dim; i++) {
        float label = labels.findActiveNeuron<LABEL_DENSE>(i).activation;
        float act = output.findActiveNeuron<DENSE>(i).activation;
        float delta = label - act;
        error += delta * delta;
      }
      return error;
    }

    // If both are sparse then we need to iterate over the nonzeros from both
    // vectors. To avoid double counting the overlapping neurons we avoid
    // computing the error while iterating over the output active_neurons, if
    // the labels also contain the same active_neuron.

    float error = 0.0;
    for (uint32_t i = 0; i < output.len; i++) {
      float label = labels.findActiveNeuron<LABEL_DENSE>(i).activation;
      if (label > 0.0) {
        continue;
      }
      float act = output.findActiveNeuron<DENSE>(i).activation;
      float delta = label - act;
      error += delta * delta;
    }

    for (uint32_t i = 0; i < labels.len; i++) {
      float label = labels.findActiveNeuron<LABEL_DENSE>(i).activation;
      float act = output.findActiveNeuron<DENSE>(i).activation;
      float delta = label - act;
      error += delta * delta;
    }
    return error;
  }

  std::atomic<float> _mse;
  std::atomic<uint32_t> _num_samples;
};

/**
 * The weighted mean absolute percentage error is a regression error that
 * measures the absolute deviation of predictions from the true values, weighted
 * in proportion to the true values. WMAPE = sum(|actual - prediction|) /
 * sum(|actual|) Here, the actual value is assumed to be non-negative. The
 * returned metric is in absolute terms; 1.0 is 100%.
 */
class WeightedMeanAbsolutePercentageError final : public Metric {
 public:
  WeightedMeanAbsolutePercentageError()
      : _sum_of_deviations(0.0), _sum_of_truths(0.0) {}

  void computeMetric(const BoltVector& output, const BoltVector& labels) final {
    // Calculate |actual - predicted| and |actual|.
    float sum_of_squared_differences = 0.0;
    float sum_of_squared_label_elems = 0.0;
    MetricUtilities::visitActiveNeurons(
        output, labels, [&](float label_val, float output_val) {
          float difference = label_val - output_val;
          sum_of_squared_differences += difference * difference;
          sum_of_squared_label_elems += label_val * label_val;
        });

    // Add to respective atomic accumulators
    MetricUtilities::incrementAtomicFloat(
        _sum_of_deviations, std::sqrt(sum_of_squared_differences));
    MetricUtilities::incrementAtomicFloat(
        _sum_of_truths, std::sqrt(sum_of_squared_label_elems));
  }

  double getMetricAndReset(bool verbose) final {
    double wmape = _sum_of_deviations /
                   std::max(_sum_of_truths.load(std::memory_order_relaxed),
                            std::numeric_limits<float>::epsilon());
    if (verbose) {
      std::cout << "Weighted Mean Absolute Percentage Error: "
                << std::setprecision(3) << wmape << " (" << wmape * 100 << "%)"
                << std::endl;
    }
    _sum_of_deviations = 0.0;
    _sum_of_truths = 0.0;
    return wmape;
  }

  static constexpr const char* name = "weighted_mean_absolute_percentage_error";

  std::string getName() final { return name; }

 private:
  std::atomic<float> _sum_of_deviations;
  std::atomic<float> _sum_of_truths;
};

/**
 * The F-Measure is a metric that takes into account both precision and recall.
 * It is defined as the harmonic mean of precision and recall. The returned
 * metric is in absolute terms; 1.0 is 100%.
 */
class FMeasure final : public Metric {
 public:
  explicit FMeasure(float threshold)
      : _threshold(threshold),
        _true_positive(0),
        _false_positive(0),
        _false_negative(0) {}

  void computeMetric(const BoltVector& output, const BoltVector& labels) final {
    auto predictions = output.getThresholdedNeurons(
        /* activation_threshold = */ _threshold,
        /* return_at_least_one = */ true,
        /* max_count_to_return = */ std::numeric_limits<uint32_t>::max());

    for (uint32_t pred : predictions) {
      if (labels.findActiveNeuronNoTemplate(pred).activation > 0) {
        _true_positive++;
      } else {
        _false_positive++;
      }
    }

    for (uint32_t pos = 0; pos < labels.len; pos++) {
      uint32_t label_active_neuron =
          labels.isDense() ? pos : labels.active_neurons[pos];
      if (labels.findActiveNeuronNoTemplate(label_active_neuron).activation >
          0) {
        if (std::find(predictions.begin(), predictions.end(),
                      label_active_neuron) == predictions.end()) {
          _false_negative++;
        }
      }
    }
  }

  double getMetricAndReset(bool verbose) final {
    double prec = static_cast<double>(_true_positive) /
                  (_true_positive + _false_positive);
    double recall = static_cast<double>(_true_positive) /
                    (_true_positive + _false_negative);
    double f_measure;

    if (prec == 0 && recall == 0) {
      f_measure = 0;
    } else {
      f_measure = (2 * prec * recall) / (prec + recall);
    }

    if (verbose) {
      std::cout << "Precision (t=" << _threshold << "): " << prec << std::endl;
      std::cout << "Recall (t=" << _threshold << "): " << recall << std::endl;
      std::cout << "F-Measure (t=" << _threshold << "): " << f_measure
                << std::endl;
    }
    _true_positive = 0;
    _false_positive = 0;
    _false_negative = 0;
    return f_measure;
  }

  static constexpr const char* name = "f_measure";

  std::string getName() final {
    std::stringstream name_ss;
    name_ss << name << '(' << _threshold << ')';
    return name_ss.str();
  }

  static bool isFMeasure(const std::string& name) {
    return std::regex_match(name, std::regex("f_measure\\(0\\.\\d+\\)"));
  }

  static std::shared_ptr<Metric> make(const std::string& name) {
    std::string token = name.substr(name.find('('));  // token = (X.XXX)
    token = token.substr(1, token.length() - 2);      // token = X.XXX
    float threshold = std::stof(token);
    return std::make_shared<FMeasure>(threshold);
  }

 private:
  float _threshold;
  std::atomic<uint64_t> _true_positive;
  std::atomic<uint64_t> _false_positive;
  std::atomic<uint64_t> _false_negative;
};

}  // namespace thirdai::bolt