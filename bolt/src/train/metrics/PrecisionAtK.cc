#include "PrecisionAtK.h"

namespace thirdai::bolt::train::metrics {

PrecisionAtK::PrecisionAtK(nn::autograd::ComputationPtr outputs,
                           nn::autograd::ComputationPtr labels, uint32_t k)
    : ComparativeMetric(std::move(outputs), std::move(labels)),
      _num_correct_predicted(0),
      _num_predicted(0),
      _k(k) {}

void PrecisionAtK::record(const BoltVector& output, const BoltVector& label) {
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

}  // namespace thirdai::bolt::train::metrics