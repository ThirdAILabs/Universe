#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/metrics/ComparativeMetric.h>
#include <bolt_vector/src/BoltVector.h>
#include <atomic>

namespace thirdai::bolt::train::metrics {

/**
 * Computes the categorical accuracy (precision@1) for the given output.
 */
class CategoricalAccuracy final : public ComparativeMetric {
 public:
  CategoricalAccuracy(nn::autograd::ComputationPtr outputs,
                      nn::autograd::ComputationPtr labels);

  void record(const BoltVector& output, const BoltVector& label) final;

  void reset() final;

  float value() const final;

  float worst() const final;

  bool betterThan(float a, float b) const final;

 private:
  std::atomic_uint64_t _correct;
  std::atomic_uint64_t _num_samples;
};

}  // namespace thirdai::bolt::train::metrics