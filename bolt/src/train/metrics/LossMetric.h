#pragma once

#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/metrics/Metric.h>
#include <atomic>

namespace thirdai::bolt::train::metrics {

/**
 * Metric representing the loss of a given loss function.
 */
class LossMetric final : public Metric {
 public:
  explicit LossMetric(nn::loss::LossPtr loss_fn);

  void record(uint32_t index_in_batch) final;

  void reset() final;

  float value() const final;

  float worst() const final;

  bool betterThan(float a, float b) const final;

 private:
  nn::loss::LossPtr _loss_fn;

  std::atomic<float> _loss;
  std::atomic<uint32_t> _num_samples;
};

}  // namespace thirdai::bolt::train::metrics