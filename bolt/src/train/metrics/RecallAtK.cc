#include "RecallAtK.h"
#include <bolt_vector/src/BoltVector.h>
#include <atomic>

namespace thirdai::bolt::train::metrics {

RecallAtK::RecallAtK(nn::autograd::ComputationPtr outputs,
                     nn::autograd::ComputationPtr labels, uint32_t k)
    : _outputs(std::move(outputs)),
      _labels(std::move(labels)),
      _true_positives(0),
      _total_positives(0),
      _k(k) {}

void RecallAtK::record(uint32_t index_in_batch) {
  const BoltVector& output = _outputs->tensor()->getVector(index_in_batch);
  const BoltVector& label = _labels->tensor()->getVector(index_in_batch);

  TopKActivationsQueue top_k_predictions = output.findKLargestActivations(_k);

  _true_positives += truePositivesInTopK(top_k_predictions, label);

  for (uint32_t i = 0; i < label.len; i++) {
    if (label.activations[i] > 0) {
      _total_positives++;
    }
  }
}

void RecallAtK::reset() {
  _true_positives = 0;
  _total_positives = 0;
}

float RecallAtK::value() const {
  return divideTwoAtomicIntegers(_true_positives, _total_positives);
}

float RecallAtK::worst() const { return 0.0; }

bool RecallAtK::betterThan(float a, float b) const { return a > b; }

}  // namespace thirdai::bolt::train::metrics