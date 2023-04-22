#include "RecallAtK.h"

namespace thirdai::bolt::train::metrics {

RecallAtK::RecallAtK(nn::autograd::ComputationPtr outputs,
                     nn::autograd::ComputationPtr labels, uint32_t k)
    : _outputs(std::move(outputs)),
      _labels(std::move(labels)),
      _num_correct_predicted(0),
      _num_ground_truth(0),
      _k(k) {}

void RecallAtK::record(uint32_t index_in_batch) {
  const auto& output = _outputs->tensor();
  const auto& labels = _labels->tensor();

  uint32_t start = output->rangeStart(index_in_batch);
  uint32_t end = output->rangeEnd(index_in_batch);

  for (uint32_t i = start; i < end; i++) {
    const auto& label_vec = labels->getVector(i);
    _num_correct_predicted +=
        truePositivesInTopK(output->getVector(i), label_vec, _k);

    for (uint32_t i = 0; i < label_vec.len; i++) {
      if (label_vec.activations[i] > 0) {
        _num_ground_truth++;
      }
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