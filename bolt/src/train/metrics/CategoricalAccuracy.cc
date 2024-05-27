#include "CategoricalAccuracy.h"
#include <bolt/src/train/metrics/Metric.h>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt::metrics {

CategoricalAccuracy::CategoricalAccuracy(ComputationPtr outputs,
                                         ComputationPtr labels)
    : _outputs(std::move(outputs)),
      _labels(std::move(labels)),
      _correct(0),
      _num_samples(0) {}

void CategoricalAccuracy::record(uint32_t index_in_batch) {
  const BoltVector& output = _outputs->tensor()->getVector(index_in_batch);
  const BoltVector& label = _labels->tensor()->getVector(index_in_batch);

  uint32_t prediction = output.getHighestActivationId();

  if (label.findActiveNeuronNoTemplate(prediction).activation > 0) {
    _correct++;
  }
  _num_samples++;
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

}  // namespace thirdai::bolt::metrics