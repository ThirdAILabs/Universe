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

namespace thirdai::bolt_v1 {

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

  // Returns the worst value a metric can hold. Useful to initialize a best
  // value, which is then updated from time to time.
  virtual double worst() const = 0;

  // Compare x, y and tell if x is better than y, when a metric of this class is
  // considered. Follows a convention to use non-strict better than so an update
  // at a later time-step in code run is marked as an improvement over a prior,
  // for the same value.
  virtual bool betterThan(double x, double y) const = 0;

  virtual ~Metric() = default;
};

/**
 * The categorical accuracy is the accuracy @1 which measures for what fraction
 * of the samples the neuron with the highest activation is in the labels.
 */
class CategoricalAccuracy final : public Metric {
 public:
  CategoricalAccuracy() : _correct(0), _num_samples(0) {}

  void record(const BoltVector& output, const BoltVector& labels) final;

  double value() final;

  void reset() final;

  static constexpr const char* NAME = "categorical_accuracy";

  std::string name() final { return NAME; }

  std::string summary() final;

  double worst() const final { return 0.0; }

  bool betterThan(double x, double y) const final { return x >= y; }

 private:
  std::atomic<uint32_t> _correct;
  std::atomic<uint32_t> _num_samples;
};

/**
 * The CategoricalCrossEntropy (metric) is a proxy to the
 * CategoricalCrossEntropy (LossFunction) to track the metric that is closer to
 * the training objective.
 *
 * This is a proxy and not true cross-entropy because computations involve
 * summing terms containing label * log (output). EPS value of 1e-7 is used to
 * generate a sufficiently high indicator for loss where log(0 + EPS) could
 * occur.
 */
class CategoricalCrossEntropy final : public Metric {
 public:
  CategoricalCrossEntropy() : _sum(0), _num_samples(0) {}

  void record(const BoltVector& output, const BoltVector& labels) final;

  double value() final {
    double acc = static_cast<double>(_sum) / _num_samples;
    return acc;
  }

  double worst() const final { return std::numeric_limits<float>::max(); }
  bool betterThan(double x, double y) const final { return x <= y; }

  void reset() final {
    _sum = 0;
    _num_samples = 0;
  }

  static constexpr const char* NAME = "categorical_cross_entropy";

  std::string name() final { return NAME; }

  std::string summary() final {
    return fmt::format("{}: {:.3f}", NAME, value());
  }

 private:
  std::atomic<float> _sum;
  std::atomic<uint32_t> _num_samples;
};

class MeanSquaredErrorMetric final : public Metric {
 public:
  MeanSquaredErrorMetric() : _mse(0), _num_samples(0) {}

  void record(const BoltVector& output, const BoltVector& labels) final;

  double value() final;

  void reset() final;

  static constexpr const char* NAME = "mean_squared_error";

  std::string name() final { return NAME; }

  std::string summary() final;

  double worst() const final { return std::numeric_limits<double>::max(); }

  bool betterThan(double x, double y) const final { return x <= y; }

 private:
  template <bool OUTPUT_DENSE, bool LABEL_DENSE>
  float computeMSE(const BoltVector& output, const BoltVector& labels);

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

  void record(const BoltVector& output, const BoltVector& labels) final;

  double value() final;

  void reset() final;

  static constexpr const char* NAME = "weighted_mean_absolute_percentage_error";

  std::string name() final { return NAME; }

  std::string summary() final;

  double worst() const final { return std::numeric_limits<double>::max(); }

  bool betterThan(double x, double y) const final { return x <= y; }

 private:
  std::atomic<float> _sum_of_deviations;
  std::atomic<float> _sum_of_truths;
};

class RecallAtK : public Metric {
 public:
  explicit RecallAtK(uint32_t k) : _k(k), _matches(0), _label_count(0) {}

  void record(const BoltVector& output, const BoltVector& labels) final;

  double value() final;

  void reset() final;

  std::string name() final { return "recall@" + std::to_string(_k); }

  std::string summary() final;

  double worst() const final { return 0.0F; }

  bool betterThan(double x, double y) const final { return x >= y; }

  static bool isRecallAtK(const std::string& name);

  static std::shared_ptr<Metric> make(const std::string& name);

 private:
  static uint32_t countLabels(const BoltVector& labels);

  uint32_t _k;
  std::atomic_uint64_t _matches;
  std::atomic_uint64_t _label_count;
};

class PrecisionAtK : public Metric {
 public:
  explicit PrecisionAtK(uint32_t k) : _k(k), _correct_guesses(0), _samples(0) {}

  void record(const BoltVector& output, const BoltVector& labels) final;

  double value() final;

  void reset() final;

  std::string name() final { return "precision@" + std::to_string(_k); }

  std::string summary() final;

  double worst() const final { return 0.0F; }

  bool betterThan(double x, double y) const final { return x >= y; }

  static bool isPrecisionAtK(const std::string& name);

  static std::shared_ptr<Metric> make(const std::string& name);

 private:
  uint32_t _k;
  std::atomic_uint64_t _correct_guesses;
  std::atomic_uint64_t _samples;
};

/**
 * The F-Measure is a metric that takes into account both precision and recall.
 * It is defined as the harmonic mean of precision and recall. The returned
 * metric is in absolute terms; 1.0 is 100%.
 */
class FMeasure final : public Metric {
 public:
  explicit FMeasure(float threshold, float beta = 1)
      : _beta_squared(beta * beta),
        _threshold(threshold),
        _true_positive(0),
        _false_positive(0),
        _false_negative(0) {}

  void record(const BoltVector& output, const BoltVector& labels) final;

  double value() final;

  std::tuple<double, double, double> metrics();

  void reset() final;

  static constexpr const char* NAME = "f_measure";

  std::string name() final;

  std::string summary() final;

  double worst() const final { return 0.0F; }

  bool betterThan(double x, double y) const final { return x >= y; }

  static bool isFMeasure(const std::string& name);
  static std::shared_ptr<Metric> make(const std::string& name);

 private:
  float _beta_squared;
  float _threshold;
  std::atomic<uint64_t> _true_positive;
  std::atomic<uint64_t> _false_positive;
  std::atomic<uint64_t> _false_negative;
};

std::shared_ptr<Metric> makeMetric(const std::string& name);

using MetricData = std::unordered_map<std::string, std::vector<double>>;
using InferenceMetricData = std::unordered_map<std::string, double>;

}  // namespace thirdai::bolt_v1
