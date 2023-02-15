#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/metrics/Metric.h>
#include <atomic>

namespace thirdai::bolt::train::metrics {

/**
 * Computes the categorical accuracy (precision@1) for the given output.
 */
class CategoricalAccuracy final : public Metric {
 public:
  CategoricalAccuracy(nn::autograd::ComputationPtr outputs,
                      nn::autograd::ComputationPtr labels);

  void record(uint32_t index_in_batch) final;

  void reset() final;

  float value() const final;

  float worst() const final;

  bool betterThan(float a, float b) const final;

 private:
  nn::autograd::ComputationPtr _outputs;
  nn::autograd::ComputationPtr _labels;

  std::atomic_uint32_t _correct;
  std::atomic_uint32_t _num_samples;
};

}  // namespace thirdai::bolt::train::metrics