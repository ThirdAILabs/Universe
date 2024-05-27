#include "PrecisionAtK.h"

namespace thirdai::bolt::metrics {

PrecisionAtK::PrecisionAtK(ComputationPtr outputs, ComputationPtr labels,
                           uint32_t k)
    : _outputs(std::move(outputs)),
      _labels(std::move(labels)),
      _num_correct_predicted(0),
      _num_predicted(0),
      _k(k) {}

void PrecisionAtK::record(uint32_t index_in_batch) {
  const BoltVector& output = _outputs->tensor()->getVector(index_in_batch);
  const BoltVector& label = _labels->tensor()->getVector(index_in_batch);

  _num_correct_predicted += truePositivesInTopK(output, label, _k);
  _num_predicted += _k;
}

void PrecisionAtK::reset() {
  _num_correct_predicted = 0;
  _num_predicted = 0;
}

float PrecisionAtK::value() const {
  return divideTwoAtomicIntegers(_num_correct_predicted, _num_predicted);
}

float PrecisionAtK::worst() const { return 0.0; }

bool PrecisionAtK::betterThan(float a, float b) const { return a > b; }

}  // namespace thirdai::bolt::metrics