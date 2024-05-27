#pragma once

#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/train/metrics/Metric.h>
#include <dataset/src/mach/MachIndex.h>

namespace thirdai::bolt::metrics {

/**
 * Computes the precision@k for the given output. Using the mach index to
 * determine predictions.
 *
 * Precision@k in this case is defined as the fraction of the time the correct
 * label is found in the top-k predictions returned from the model using the
 * given mach index.
 */
class MachPrecision final : public Metric {
 public:
  MachPrecision(dataset::mach::MachIndexPtr mach_index,
                uint32_t num_buckets_to_eval, ComputationPtr outputs,
                ComputationPtr labels, uint32_t k);

  void record(uint32_t index_in_batch) final;

  void reset() final;

  float value() const final;

  float worst() const final;

  bool betterThan(float a, float b) const final;

 private:
  dataset::mach::MachIndexPtr _mach_index;
  uint32_t _num_buckets_to_eval;

  ComputationPtr _outputs;
  ComputationPtr _labels;

  std::atomic_uint64_t _num_correct_predicted = 0;
  std::atomic_uint64_t _num_predicted = 0;
  uint32_t _k;
};

}  // namespace thirdai::bolt::metrics