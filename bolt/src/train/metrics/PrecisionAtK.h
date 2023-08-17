#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/metrics/Metric.h>

namespace thirdai::bolt::metrics {

/**
 * Computes the precision@k for the given output.
 *
 * Precision@k is the number of true samples in the top k items
 * divided by k
 * https://en.wikipedia.org/wiki/Precision_and_recall#Precision
 */
class PrecisionAtK final : public Metric {
 public:
  PrecisionAtK(ComputationPtr outputs, ComputationPtr labels, uint32_t k);

  void record(uint32_t index_in_batch) final;

  void reset() final;

  float value() const final;

  float worst() const final;

  bool betterThan(float a, float b) const final;

 private:
  ComputationPtr _outputs;
  ComputationPtr _labels;

  std::atomic_uint64_t _num_correct_predicted;
  std::atomic_uint64_t _num_predicted;
  uint32_t _k;
};

}  // namespace thirdai::bolt::metrics