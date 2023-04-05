#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/metrics/Metric.h>

namespace thirdai::bolt::train::metrics {

/**
 * Computes the precision@k for the given output.
 *
 * Precision@k measures the proportion of relevant items among the top k items
 * returned by a ranking algorithm or classifier. It is calculated as the number
 * of true positives among the top k predictions divided by k.
 * https://en.wikipedia.org/wiki/Precision_and_recall#Precision
 */
class PrecisionAtK final : public Metric {
 public:
  PrecisionAtK(nn::autograd::ComputationPtr outputs,
               nn::autograd::ComputationPtr labels, uint32_t k);

  void record(uint32_t index_in_batch) final;

  void reset() final;

  float value() const final;

  float worst() const final;

  bool betterThan(float a, float b) const final;

 private:
  nn::autograd::ComputationPtr _outputs;
  nn::autograd::ComputationPtr _labels;

  std::atomic_uint32_t _true_positives;
  std::atomic_uint32_t _predicted_positives;
  uint32_t _k;
};

}  // namespace thirdai::bolt::train::metrics