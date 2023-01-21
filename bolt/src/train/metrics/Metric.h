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
 * and/or validation. Metrics are bound to a particular output/label pair in the
 * model and cannot be reused once passed into the model. Currently metrics can
 * only be applied to outputs which are used uniquely in a loss function with a
 * single label. For instance if a loss function is computed on two outputs then
 * those outputs are not available for metrics.
 */
class Metric {
 public:
  /**
   * Updates the current value of the metric with the ith sample in the batch.
   * This method is automatically called in parallel accross different elements
   * of a batch and must support concurency.
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
  virtual std::string name() const = 0;

  /**
   * Binds the outputs to the metric.
   */
  virtual void setOutputs(nn::autograd::ComputationPtr outputs) = 0;

  /**
   * Binds the labels to the metric.
   */
  virtual void setLabels(nn::autograd::ComputationPtr labels) = 0;

  /**
   * Returns the name of the output the metric is bound to.
   */
  virtual std::string outputName() const = 0;

  virtual ~Metric() = default;

  /**
   * Helper method for incrementing an atomic float.
   */
  static void incrementAtomicFloat(std::atomic<float>& value, float increment);
};

using MetricPtr = std::shared_ptr<Metric>;

// Maps outputs to metrics to values.
using History =
    std::unordered_map<std::string,
                       std::unordered_map<std::string, std::vector<float>>>;

using HistoryPtr = std::shared_ptr<History>;

// How metrics are provided to the trainer. Maps output names to lists of
// metrics.
using InputMetrics =
    std::unordered_map<std::string, std::vector<metrics::MetricPtr>>;

/**
 * Represents a collection of metrics. Binds the metrics their given output and
 * label in its constructor.
 */
class MetricList {
 public:
  MetricList(const InputMetrics& metrics, const nn::model::ModelPtr& model);

  /**
   * Updates the values of the metrics based on the current batch.
   */
  void recordBatch(uint32_t batch_size);

  /**
   * Updates the given history with the values of each metric.
   */
  void updateHistory(HistoryPtr& history, const std::string& prefix);

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

}  // namespace thirdai::bolt::train::metrics