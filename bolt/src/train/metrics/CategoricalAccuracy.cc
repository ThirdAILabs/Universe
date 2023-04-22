#include "CategoricalAccuracy.h"
#include <bolt/src/train/metrics/Metric.h>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt::train::metrics {

CategoricalAccuracy::CategoricalAccuracy(nn::autograd::ComputationPtr outputs,
                                         nn::autograd::ComputationPtr labels)
    : _outputs(std::move(outputs)),
      _labels(std::move(labels)),
      _correct(0),
      _num_samples(0) {}

void CategoricalAccuracy::record(uint32_t index_in_batch) {
  const auto& output = _outputs->tensor();
  const auto& labels = _labels->tensor();

  uint32_t start = output->rangeStart(index_in_batch);
  uint32_t end = output->rangeEnd(index_in_batch);

  for (uint32_t i = start; i < end; i++) {
    uint32_t prediction = output->getVector(i).getHighestActivationId();

    if (labels->getVector(i).findActiveNeuronNoTemplate(prediction).activation >
        0) {
      _correct++;
    }
    _num_samples++;
  }
}

void CategoricalAccuracy::reset() {
  _correct = 0;
  _num_samples = 0;
}

float CategoricalAccuracy::value() const {
  return divideTwoAtomicIntegers(_correct, _num_samples);
}

float CategoricalAccuracy::worst() const { return 0.0; }

bool CategoricalAccuracy::betterThan(float a, float b) const { return a > b; }

}  // namespace thirdai::bolt::train::metrics