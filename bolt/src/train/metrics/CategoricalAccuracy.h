#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/metrics/Metric.h>
#include <atomic>

namespace thirdai::bolt::metrics {

/**
 * Computes the categorical accuracy (precision@1) for the given output.
 */
class CategoricalAccuracy final : public Metric {
 public:
  CategoricalAccuracy(ComputationPtr outputs, ComputationPtr labels);

  void record(uint32_t index_in_batch) final;

  void reset() final;

  float value() const final;

  float worst() const final;

  bool betterThan(float a, float b) const final;

 private:
  ComputationPtr _outputs;
  ComputationPtr _labels;

  std::atomic_uint64_t _correct;
  std::atomic_uint64_t _num_samples;
};

}  // namespace thirdai::bolt::metrics