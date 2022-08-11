#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/metrics/MetricHelpers.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <queue>
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
  virtual void computeMetric(const BoltVector& output,
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
          "representable float. This is likely do to a Nan or incorrect "
          "activation function in the final layer.");
    }

    // The nueron with the largest activation is the prediction
    uint32_t pred = output.isDense() ? *max_act_index
                                     : output.active_neurons[*max_act_index];

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
 * Root mean squared error (RMSE) is a standard regression metric.
 * RMSE = sqrt(sum((actual - prediction)^2))
 */
class RootMeanSquaredError final : public Metric {
 public:
  RootMeanSquaredError() : _sum_of_squared_errors(0.0), _count(0) {}

  void computeMetric(const BoltVector& output, const BoltVector& labels) final {
    float squared_errors = 0.0;
    MetricUtilities::visitActiveNeurons(output, labels,
                                        [&](float label_val, float output_val) {
                                          float error = label_val - output_val;
                                          squared_errors += error * error;
                                        });

    // Add to respective atomic accumulators
    MetricUtilities::incrementAtomicFloat(_sum_of_squared_errors,
                                          squared_errors);

    _count.fetch_add(1);
  }

  double getMetricAndReset(bool verbose) final {
    double rmse = std::sqrt(_sum_of_squared_errors / _count);
    if (verbose) {
      std::cout << "Root Mean Squared Error: " << std::setprecision(3) << rmse
                << std::endl;
    }
    _sum_of_squared_errors = 0.0;
    _count = 0;
    return rmse;
  }

  static constexpr const char* name = "root_mean_squared_error";

  std::string getName() final { return name; }

 private:
  std::atomic<float> _sum_of_squared_errors;
  std::atomic<uint64_t> _count;
};

class PrecisionAt : public Metric {
 public:
  explicit PrecisionAt(uint32_t k): _k(k), _hits(0), _correct(0), _sample_count(0), _retrieved_count(0), _label_count(0) {}

  void computeMetric(const BoltVector& output, const BoltVector& labels) final {
    auto top_k = output.isDense() 
        ? topK</* DENSE= */true>(output, _k) 
        : topK</* DENSE= */false>(output, _k);
    auto correct = labels.isDense()
        ? countCorrectInTopK</* DENSE= */ true>(std::move(top_k), labels)
        : countCorrectInTopK</* DENSE= */ false>(std::move(top_k), labels);
    
    _correct.fetch_add(correct);
    if (correct > 0) {
      _hits.fetch_add(1);
    }
    _sample_count.fetch_add(1);
    _retrieved_count.fetch_add(_k);
    _label_count.fetch_add(labels.len);
  }

  double getMetricAndReset(bool verbose) final {
    std::cout << "Precision at " << _k << ": " << static_cast<double>(_correct) / _retrieved_count << std::endl;
    std::cout << "Recall at " << _k << ": " << static_cast<double>(_correct) / _label_count << std::endl;
    std::cout << "Categorical accuracy at " << _k << ": " << static_cast<double>(_hits) / _sample_count << std::endl;
    
    double metric = static_cast<double>(_hits) / _sample_count;
    if (verbose) {
      std::cout << "Hit ratio at " << _k << ": " << std::setprecision(3) << metric
                << std::endl;
    }
    _correct = 0;
    _hits = 0;
    _sample_count = 0;
    _retrieved_count = 0;
    _label_count = 0;
    return metric;
  }

  static constexpr const char* name = "hit_ratio_at_";

  std::string getName() final { 
    std::stringstream name_ss;
    name_ss << name << _k;
    return name_ss.str(); 
  }

  static inline bool isPrecisionAtK(const std::string& metric_name) {
    return metric_name.substr(0, 13) == name;
  }

  static inline uint32_t getK(const std::string& metric_name) {
    auto k = metric_name.substr(13);
    char* end_ptr;
    return std::strtol(k.data(), &end_ptr, 10);
  }
 
 private:
  using val_idx_pair_t = std::pair<float, uint32_t>; 
  using top_k_t = std::priority_queue<val_idx_pair_t, std::vector<val_idx_pair_t>, std::greater<val_idx_pair_t>>;

  template<bool DENSE>
  static inline top_k_t topK(const BoltVector& output, uint32_t k) {
    top_k_t top_k;
    for (uint32_t pos = 0; pos < std::min(k, output.len); pos++) {
      top_k.push(std::move(valueIndexPair<DENSE>(output, pos)));
    }
    for (uint32_t pos = k; pos < output.len; pos++) {
      auto val_idx_pair = valueIndexPair<DENSE>(output, pos);
      if (val_idx_pair > top_k.top()) {
        top_k.pop();
        top_k.push(std::move(val_idx_pair));
      }
    }
    return top_k;
  }

  template<bool DENSE>
  static inline val_idx_pair_t valueIndexPair(const BoltVector& output, uint32_t pos) {
    if (DENSE) {
      return {output.activations[pos], pos};
    }
    return {output.activations[pos], output.active_neurons[pos]};
  }

  template<bool DENSE>
  static inline uint32_t countCorrectInTopK(top_k_t&& top_k, const BoltVector& labels) {
    uint32_t correct = 0;
    for (uint32_t i = 0; i < top_k.size(); i++) {
      if (labels.findActiveNeuron<DENSE>(/* active_neuron= */ top_k.top().second).activation > 0) {
        correct++;
      }
      top_k.pop();
    }
    return correct;
  }

  uint32_t _k;
  std::atomic_uint64_t _hits;
  std::atomic_uint64_t _correct;
  std::atomic_uint64_t _sample_count;
  std::atomic_uint64_t _retrieved_count;
  std::atomic_uint64_t _label_count;
};

using MetricData = std::unordered_map<std::string, std::vector<double>>;
using InferenceMetricData = std::unordered_map<std::string, double>;

}  // namespace thirdai::bolt