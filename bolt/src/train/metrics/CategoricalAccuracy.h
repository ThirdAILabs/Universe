#pragma once

#include <bolt/src/train/metrics/ComparativeMetric.h>
#include <atomic>

namespace thirdai::bolt::train::metrics {

/**
 * Computes the categorical accuracy (precision@1) for the given output.
 */
class CategoricalAccuracy final : public ComparativeMetric {
 public:
  CategoricalAccuracy();

  void reset() final;

  float value() const final;

  float worst() const final;

  bool betterThan(float a, float b) const final;

  std::string name() const final;

 protected:
  void record(const BoltVector& output, const BoltVector& label) final;

 private:
  std::atomic_uint32_t _correct;
  std::atomic_uint32_t _num_samples;
};

}  // namespace thirdai::bolt::train::metrics