#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace thirdai::bolt::train::metrics {

/**
 * Metrics represent some value or measure that is computed during training
 * and/or validation. The metric should be constructed with computations it
 * needs to access to compute the metric.
 */
class Metric {
 public:
  /**
   * Updates the current value of the metric with the ith sample in the batch.
   * This method is automatically called in parallel accross different elements
   * of a batch and must support concurency. This should use the computations
   * the metric was constructed with.
   */
  virtual void record(uint32_t index_in_batch) = 0;

  /**
   * Resets the value of the metric.
   */
  virtual void reset() = 0;

  /**
   * Returns the value held in the metric.
   */
  virtual float value() const = 0;

  /**
   * Returns the worst possible value of the metric.
   */
  virtual float worst() const = 0;

  /**
   * Returns if the value a is better than the value b for the given metric.
   */
  virtual bool betterThan(float a, float b) const = 0;

  /**
   * Returns the name of the metric.
   */
  std::string name() const { return _name; }

  void setName(const std::string& name) { _name = name; }

  virtual ~Metric() = default;

  /**
   * Helper method for incrementing an atomic float.
   */
  static void incrementAtomicFloat(std::atomic<float>& value, float increment);

 private:
  std::string _name;
};

using MetricPtr = std::shared_ptr<Metric>;

// Maps metric names of each metric to all the values that been computed for
// that metric. Contains values from both training and validation metrics. Names
// are supplied by the user when initially specifying the metrics.
using History = std::unordered_map<std::string, std::vector<float>>;

using HistoryPtr = std::shared_ptr<History>;

// How metrics are provided to the trainer.
using InputMetrics = std::unordered_map<std::string, metrics::MetricPtr>;

/**
 * Represents a collection of metrics.
 */
class MetricCollection {
 public:
  explicit MetricCollection(const InputMetrics& metrics);

  /**
   * Updates the values of the metrics based on the current batch.
   */
  void recordBatch(uint32_t batch_size);

  /**
   * Updates the given history with the values of each metric. The metrics store
   * there values in the history under the key <prefix>_<metric_name>. This is
   * used to distinguish train vs validation metrics.
   */
  void updateHistory(History& history);

  /**
   * Creates a string summary of the current values of the metrics.
   */
  std::string summarizeLastStep() const;

  /**
   * Resets all metrics.
   */
  void reset();

 private:
  std::vector<MetricPtr> _metrics;
};

InputMetrics metricsForSingleOutputModel(
    const std::vector<std::string>& metric_names,
    const nn::autograd::ComputationPtr& output,
    const nn::autograd::ComputationPtr& labels);

}  // namespace thirdai::bolt::train::metrics