#include "PrecisionAtK.h"
#include <bolt_vector/src/BoltVector.h>
#include <atomic>

namespace thirdai::bolt::train::metrics {

PrecisionAtK::PrecisionAtK(nn::autograd::ComputationPtr outputs,
                           nn::autograd::ComputationPtr labels, uint32_t k)
    : _outputs(std::move(outputs)),
      _labels(std::move(labels)),
      _true_positives(0),
      _predicted_positives(0),
      _k(k) {}

void PrecisionAtK::record(uint32_t index_in_batch) {
  const BoltVector& output = _outputs->tensor()->getVector(index_in_batch);
  const BoltVector& label = _labels->tensor()->getVector(index_in_batch);

  TopKActivationsQueue top_k_predictions = output.findKLargestActivations(_k);

  _true_positives += truePositivesInTopK(top_k_predictions, label);
  _predicted_positives += _k;
}

void PrecisionAtK::reset() {
  _true_positives = 0;
  _predicted_positives = 0;
}

float PrecisionAtK::value() const {
  return divideTwoAtomicIntegers(_true_positives, _predicted_positives);
}

float PrecisionAtK::worst() const { return 0.0; }

bool PrecisionAtK::betterThan(float a, float b) const { return a > b; }

}  // namespace thirdai::bolt::train::metrics