#pragma once

#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/train/metrics/Metric.h>
#include <dataset/src/mach/MachIndex.h>

namespace thirdai::bolt::train::metrics {

/**
 * Computes the precision@k for the given output. Using the mach index to
 * determine predictions.
 *
 * Precision@k in this case is defined as the fraction of the time the correct
 * label is found in the top-k predictions returned from the model using the
 * given mach index.
 */
class MachRecall final : public Metric {
 public:
  MachRecall(dataset::mach::MachIndexPtr mach_index,
             uint32_t top_k_per_eval_aggregation,
             nn::autograd::ComputationPtr outputs,
             nn::autograd::ComputationPtr labels, uint32_t k);

  void record(uint32_t index_in_batch) final;

  void reset() final;

  float value() const final;

  float worst() const final;

  bool betterThan(float a, float b) const final;

 private:
  dataset::mach::MachIndexPtr _mach_index;
  uint32_t _top_k_per_eval_aggregation;

  nn::autograd::ComputationPtr _outputs;
  nn::autograd::ComputationPtr _labels;

  std::atomic_uint64_t _num_correct_predicted = 0;
  std::atomic_uint64_t _num_ground_truth = 0;
  uint32_t _k;
};

}  // namespace thirdai::bolt::train::metrics