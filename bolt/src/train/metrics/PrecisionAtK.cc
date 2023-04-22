#include "PrecisionAtK.h"

namespace thirdai::bolt::train::metrics {

PrecisionAtK::PrecisionAtK(nn::autograd::ComputationPtr outputs,
                           nn::autograd::ComputationPtr labels, uint32_t k)
    : _outputs(std::move(outputs)),
      _labels(std::move(labels)),
      _num_correct_predicted(0),
      _num_predicted(0),
      _k(k) {}

void PrecisionAtK::record(uint32_t index_in_batch) {
  const auto& output = _outputs->tensor();
  const auto& labels = _labels->tensor();

  uint32_t start = output->rangeStart(index_in_batch);
  uint32_t end = output->rangeEnd(index_in_batch);

  for (uint32_t i = start; i < end; i++) {
    _num_correct_predicted +=
        truePositivesInTopK(output->getVector(i), labels->getVector(i), _k);
    _num_predicted += _k;
  }
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

}  // namespace thirdai::bolt::train::metrics