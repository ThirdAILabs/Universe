#include "RecallAtK.h"

namespace thirdai::bolt::metrics {

RecallAtK::RecallAtK(ComputationPtr outputs, ComputationPtr labels, uint32_t k)
    : _outputs(std::move(outputs)),
      _labels(std::move(labels)),
      _num_correct_predicted(0),
      _num_ground_truth(0),
      _k(k) {}

void RecallAtK::record(uint32_t index_in_batch) {
  const BoltVector& output = _outputs->tensor()->getVector(index_in_batch);
  const BoltVector& label = _labels->tensor()->getVector(index_in_batch);

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

}  // namespace thirdai::bolt::metrics