#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/metrics/ComparativeMetric.h>
#include <bolt/src/train/metrics/Metric.h>

namespace thirdai::bolt::train::metrics {

/**
 * Computes the recall@k for the given output.
 *
 * Recall@k is the number of true samples in the top k items
 * divided by the total number of true samples
 * https://en.wikipedia.org/wiki/Precision_and_recall#Recall
 */
class RecallAtK final : public ComparativeMetric {
 public:
  RecallAtK(nn::autograd::ComputationPtr outputs,
            nn::autograd::ComputationPtr labels, uint32_t k);

  void record(const BoltVector& output, const BoltVector& label) final;

  void reset() final;

  float value() const final;

  float worst() const final;

  bool betterThan(float a, float b) const final;

 private:
  nn::autograd::ComputationPtr _outputs;
  nn::autograd::ComputationPtr _labels;

  std::atomic_uint64_t _num_correct_predicted;
  std::atomic_uint64_t _num_ground_truth;
  uint32_t _k;
};

}  // namespace thirdai::bolt::train::metrics