#pragma once

#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/metrics/Metric.h>
#include <atomic>

namespace thirdai::bolt::metrics {

/**
 * Metric representing the loss of a given loss function.
 */
class LossMetric final : public Metric {
 public:
  explicit LossMetric(LossPtr loss_fn);

  void record(uint32_t index_in_batch) final;

  void reset() final;

  float value() const final;

  float worst() const final;

  bool betterThan(float a, float b) const final;

 private:
  LossPtr _loss_fn;

  std::atomic<float> _loss;
  std::atomic<uint32_t> _num_samples;
};

}  // namespace thirdai::bolt::metrics