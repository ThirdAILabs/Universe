#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/metrics/Metric.h>
#include <atomic>

namespace thirdai::bolt::train::metrics {

/**
 * Computes the recall@k for the given output.
 *
 * Recall@k measures the proportion of relevant items retrieved among the top k
 * items returned by a ranking algorithm or classifier. It is calculated as the
 * number of true positives among the top k predictions divided by the total
 * number of positive samples.
 * https://en.wikipedia.org/wiki/Precision_and_recall#Recall
 */
class RecallAtK final : public Metric {
 public:
  RecallAtK(nn::autograd::ComputationPtr outputs,
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
  std::atomic_uint32_t _total_positives;
  uint32_t _k;
};

}  // namespace thirdai::bolt::train::metrics