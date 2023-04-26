#include "RecallAtK.h"

namespace thirdai::bolt::train::metrics {

RecallAtK::RecallAtK(nn::autograd::ComputationPtr outputs,
                     nn::autograd::ComputationPtr labels, uint32_t k)
    : ComparativeMetric(std::move(outputs), std::move(labels)),
      _num_correct_predicted(0),
      _num_ground_truth(0),
      _k(k) {}

void RecallAtK::record(const BoltVector& output, const BoltVector& label) {
  _num_correct_predicted += truePositivesInTopK(output, label, _k);

  for (uint32_t i = 0; i < label.len; i++) {
    if (label.activations[i] > 0) {
      _num_ground_truth++;
    }
  }
}

void RecallAtK::reset() {
  _num_correct_predicted = 0;
  _num_ground_truth = 0;
}

float RecallAtK::value() const {
  return divideTwoAtomicIntegers(_num_correct_predicted, _num_ground_truth);
}

float RecallAtK::worst() const { return 0.0; }

bool RecallAtK::betterThan(float a, float b) const { return a > b; }

}  // namespace thirdai::bolt::train::metrics