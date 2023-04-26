#pragma once

#include <bolt/src/train/metrics/ComparativeMetric.h>
#include <bolt/src/train/metrics/Metric.h>
#include <atomic>

namespace thirdai::bolt::train::metrics {

class FMeasure final : public ComparativeMetric {
 public:
  explicit FMeasure(nn::autograd::ComputationPtr outputs,
                    nn::autograd::ComputationPtr labels, float threshold,
                    float beta = 1);

  void record(const BoltVector& output, const BoltVector& label) final;

  void reset() final;

  float value() const final;

  float worst() const final;

  bool betterThan(float a, float b) const final;

 private:
  nn::autograd::ComputationPtr _outputs;
  nn::autograd::ComputationPtr _labels;

  std::atomic_uint64_t _true_positives;
  std::atomic_uint64_t _false_positives;
  std::atomic_uint64_t _false_negatives;

  float _threshold;
  float _beta_squared;
};

}  // namespace thirdai::bolt::train::metrics