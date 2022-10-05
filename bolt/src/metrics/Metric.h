#pragma once
#include <bolt/src/metrics/MetricHelpers.h>
#include <bolt_vector/src/BoltVector.h>
#include <sys/types.h>
#include <utils/Logging.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace thirdai::bolt {

// Metric interface
class Metric {
 public:
  // Computes and updates the value of the metric given the sample.
  // For instance this may update the accuracy.
  virtual void record(const BoltVector& output, const BoltVector& labels) = 0;

  // Gets the value of the scalar tracked by this metric.
  virtual double value() = 0;

  // Resets the metric.
  virtual void reset() = 0;

  // Summarizes the metric as a string
  virtual std::string summary() = 0;

  // Returns the name of the metric
  virtual std::string name() = 0;

  // returns whether its better if the metric is smaller. for example, with a
  // an accuracy related metric this would return false since larger is better
  // (larger means more accurate)
  virtual bool smallerIsBetter() const = 0;

  virtual ~Metric() = default;
};

/**
 * The categorical accuracy is the accuracy @1 which measures for what fraction
 * of the samples the neuron with the highest activation is in the labels.
 */
class CategoricalAccuracy final : public Metric {
 public:
  CategoricalAccuracy() : _correct(0), _num_samples(0) {}

  void record(const BoltVector& output, const BoltVector& labels) final {
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

  double value() final {
    double acc = static_cast<double>(_correct) / _num_samples;
    return acc;
  }

  void reset() final {
    _correct = 0;
    _num_samples = 0;
  }

  static constexpr const char* NAME = "categorical_accuracy";

  std::string name() final { return NAME; }

  std::string summary() final {
    std::stringstream stream;
    stream << NAME << ": " << value();
    return stream.str();
  }

  bool smallerIsBetter() const final { return false; }

 private:
  std::atomic<uint32_t> _correct;
  std::atomic<uint32_t> _num_samples;
};

/**
 * The CategoricalCrossEntropy (metric) is a proxy to the
 * CategoricalCrossEntropy (LossFunction) to track the metric that is closer to
 * the training objective.
 */
class CategoricalCrossEntropy final : public Metric {
 public:
  CategoricalCrossEntropy() : _sum(0), _num_samples(0) {}

  void record(const BoltVector& output, const BoltVector& labels) final {
    float sample_loss = 0;
    if (output.isDense()) {
      if (labels.isDense()) {
        // (Dense Output, Dense Labels)
        // If both are dense, they're expected to have the same length.
        // In this case, we may simply run over the dense vectors and compute
        // sum((p_i)log(q_i)).
        assert(output.len == labels.len);
        for (uint32_t i = 0; i < output.len; i++) {
          sample_loss +=
              labels.activations[i] * std::log(output.activations[i]);
        }
      } else {
        // (Dense Output, Sparse Labels)
        for (uint32_t i = 0; i < output.len; i++) {
          const uint32_t* label_start = labels.active_neurons;
          const uint32_t* label_end = labels.active_neurons + labels.len;

          // Find the position of the active neuron if it exists in the labels.
          const uint32_t* label_query = std::find(label_start, label_end, i);

          if (label_query != label_end) {
            // In this case, we have found the labels. Other label activations
            // are 0, so we can ignore (0*log(whatever)).
            //
            // Compute label_index to lookup the value from labels
            // sparse-vector.
            size_t label_index = std::distance(label_start, label_query);

            sample_loss += labels.activations[label_index] *
                           std::log(output.activations[i]);
          }
        }
      }
    } else {
      std::cerr << "Not implemented yet" << std::endl;
      std::abort();
    }

    // This is dangerous, unfortunately. Let's get compile working short term.
    _sum.store(_sum.load() + -1 * sample_loss);
    _num_samples++;
  }

  double value() final {
    double acc = static_cast<double>(_sum) / _num_samples;
    return acc;
  }

  void reset() final {
    _sum = 0;
    _num_samples = 0;
  }

  static constexpr const char* NAME = "xent";

  std::string name() final { return NAME; }

  std::string summary() final {
    return fmt::format("{}: {:.3f}", NAME, value());
  }

  bool smallerIsBetter() const final { return false; }

 private:
  std::atomic<float> _sum;
  std::atomic<uint32_t> _num_samples;
};

class MeanSquaredErrorMetric final : public Metric {
 public:
  MeanSquaredErrorMetric() : _mse(0), _num_samples(0) {}

  void record(const BoltVector& output, const BoltVector& labels) final {
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

  double value() final {
    double error = _mse / _num_samples;
    return error;
  }

  void reset() final {
    _mse = 0;
    _num_samples = 0;
  }

  static constexpr const char* NAME = "mean_squared_error";

  std::string name() final { return NAME; }

  std::string summary() final {
    std::stringstream stream;
    stream << NAME << ": " << value();
    return stream.str();
  }

  bool smallerIsBetter() const final { return true; }

 private:
  template <bool OUTPUT_DENSE, bool LABEL_DENSE>
  float computeMSE(const BoltVector& output, const BoltVector& labels) {
    if constexpr (OUTPUT_DENSE || LABEL_DENSE) {
      // If either vector is dense then we need to iterate over the full
      // dimension from the layer.
      uint32_t dim = std::max(output.len, labels.len);

      float error = 0.0;
      for (uint32_t i = 0; i < dim; i++) {
        float label = labels.findActiveNeuron<LABEL_DENSE>(i).activation;
        float act = output.findActiveNeuron<OUTPUT_DENSE>(i).activation;
        float delta = label - act;
        error += delta * delta;
      }
      return error;
    }

    // If both are sparse then we need to iterate over the nonzeros from both
    // vectors. To avoid double counting the overlapping neurons we avoid
    // computing the error while iterating over the labels for neurons that are
    // also in the output active neurons.
    float error = 0.0;
    for (uint32_t i = 0; i < output.len; i++) {
      float label =
          labels.findActiveNeuron<LABEL_DENSE>(output.active_neurons[i])
              .activation;
      float act = output.activations[i];
      float delta = label - act;
      error += delta * delta;
    }

    for (uint32_t i = 0; i < labels.len; i++) {
      auto output_neuron =
          output.findActiveNeuron<OUTPUT_DENSE>(labels.active_neurons[i]);
      // Skip any neurons that were in the active neuron set since the loss was
      // already computed for them.
      if (!output_neuron.pos) {
        float label = labels.activations[i];
        // The activation is 0 since this isn't in the output active neurons.
        error += label * label;
      }
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

  void record(const BoltVector& output, const BoltVector& labels) final {
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

  double value() final {
    double wmape = _sum_of_deviations /
                   std::max(_sum_of_truths.load(std::memory_order_relaxed),
                            std::numeric_limits<float>::epsilon());
    return wmape;
  }

  void reset() final {
    _sum_of_deviations = 0.0;
    _sum_of_truths = 0.0;
  }

  static constexpr const char* NAME = "weighted_mean_absolute_percentage_error";

  std::string name() final { return NAME; }

  std::string summary() final {
    std::stringstream stream;
    stream << NAME << ": " << value();
    return stream.str();
  }

  bool smallerIsBetter() const final { return true; }

 private:
  std::atomic<float> _sum_of_deviations;
  std::atomic<float> _sum_of_truths;
};

class RecallAtK : public Metric {
 public:
  explicit RecallAtK(uint32_t k) : _k(k), _matches(0), _label_count(0) {}

  void record(const BoltVector& output, const BoltVector& labels) final {
    auto top_k = output.findKLargestActivations(_k);

    uint32_t matches = 0;
    while (!top_k.empty()) {
      if (labels
              .findActiveNeuronNoTemplate(
                  /* active_neuron= */ top_k.top().second)
              .activation > 0) {
        matches++;
      }
      top_k.pop();
    }

    _matches.fetch_add(matches);
    _label_count.fetch_add(countLabels(labels));
  }

  double value() final {
    double metric = static_cast<double>(_matches) / _label_count;
    return metric;
  }

  void reset() final {
    _matches = 0;
    _label_count = 0;
  }

  std::string name() final { return "recall@" + std::to_string(_k); }

  std::string summary() final {
    std::stringstream stream;
    stream << "Recall@" << _k << ": " << std::setprecision(3) << value();
    return stream.str();
  }

  bool smallerIsBetter() const final { return false; }

  static inline bool isRecallAtK(const std::string& name) {
    return std::regex_match(name, std::regex("recall@[1-9]\\d*"));
  }

  static std::shared_ptr<Metric> make(const std::string& name) {
    if (!isRecallAtK(name)) {
      std::stringstream error_ss;
      error_ss << "Invoked RecallAtK::make with invalid string '" << name
               << "'. RecallAtK::make should be invoked with a string in "
                  "the format 'recall@k', where k is a positive integer.";
      throw std::invalid_argument(error_ss.str());
    }

    char* end_ptr;
    auto k = std::strtol(name.data() + 7, &end_ptr, 10);
    if (k <= 0) {
      std::stringstream error_ss;
      error_ss << "RecallAtK invoked with k = " << k
               << ". k should be greater than 0.";
      throw std::invalid_argument(error_ss.str());
    }

    return std::make_shared<RecallAtK>(k);
  }

 private:
  static uint32_t countLabels(const BoltVector& labels) {
    uint32_t correct_labels = 0;
    for (uint32_t i = 0; i < labels.len; i++) {
      if (labels.activations[i] > 0) {
        correct_labels++;
      }
    }
    return correct_labels;
  }

  uint32_t _k;
  std::atomic_uint64_t _matches;
  std::atomic_uint64_t _label_count;
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

  void record(const BoltVector& output, const BoltVector& labels) final {
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

  double value() final {
    auto [precision, recall, f_measure] = metrics();
    return f_measure;
  }

  std::tuple<double, double, double> metrics() {
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

    return {prec, recall, f_measure};
  }

  void reset() final {
    _true_positive = 0;
    _false_positive = 0;
    _false_negative = 0;
  }

  static constexpr const char* NAME = "f_measure";

  std::string name() final {
    std::stringstream name_ss;
    name_ss << NAME << '(' << _threshold << ')';
    return name_ss.str();
  }

  std::string summary() final {
    auto [precision, recall, f_measure] = metrics();
    std::stringstream stream;
    stream << "precision(t=" << _threshold << "):" << precision;
    stream << ", "
           << "recall(t=" << _threshold << "):" << recall;
    stream << ", "
           << "f-measure(t=" << _threshold << "):" << f_measure;
    return stream.str();
  }

  bool smallerIsBetter() const final { return false; }

  static bool isFMeasure(const std::string& name) {
    return std::regex_match(name, std::regex(R"(f_measure\(0\.\d+\))"));
  }

  static std::shared_ptr<Metric> make(const std::string& name) {
    if (!isFMeasure(name)) {
      std::stringstream error_ss;
      error_ss << "Invoked FMeasure::make with invalid string '" << name
               << "'. FMeasure::make should be invoked with a string "
                  "in the format 'f_measure(threshold)', where "
                  "threshold is a positive floating point number.";
      throw std::invalid_argument(error_ss.str());
    }

    std::string token = name.substr(name.find('('));  // token = (X.XXX)
    token = token.substr(1, token.length() - 2);      // token = X.XXX
    float threshold = std::stof(token);

    if (threshold <= 0) {
      std::stringstream error_ss;
      error_ss << "FMeasure invoked with threshold = " << threshold
               << ". The threshold should be greater than 0.";
      throw std::invalid_argument(error_ss.str());
    }

    return std::make_shared<FMeasure>(threshold);
  }

 private:
  float _threshold;
  std::atomic<uint64_t> _true_positive;
  std::atomic<uint64_t> _false_positive;
  std::atomic<uint64_t> _false_negative;
};

static std::shared_ptr<Metric> makeMetric(const std::string& name) {
  if (name == CategoricalAccuracy::NAME) {
    return std::make_shared<CategoricalAccuracy>();
  }
  if (name == WeightedMeanAbsolutePercentageError::NAME) {
    return std::make_shared<WeightedMeanAbsolutePercentageError>();
  }
  if (name == MeanSquaredErrorMetric::NAME) {
    return std::make_shared<MeanSquaredErrorMetric>();
  }
  if (FMeasure::isFMeasure(name)) {
    return FMeasure::make(name);
  }
  if (RecallAtK::isRecallAtK(name)) {
    return RecallAtK::make(name);
  }
  if (name == CategoricalCrossEntropy::NAME) {
    return std::make_shared<CategoricalCrossEntropy>();
  }
  throw std::invalid_argument("'" + name + "' is not a valid metric.");
}

using MetricData = std::unordered_map<std::string, std::vector<double>>;
using InferenceMetricData = std::unordered_map<std::string, double>;

}  // namespace thirdai::bolt
