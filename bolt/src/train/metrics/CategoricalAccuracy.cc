#include "CategoricalAccuracy.h"
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt::train::metrics {

CategoricalAccuracy::CategoricalAccuracy(nn::autograd::ComputationPtr outputs,
                                         nn::autograd::ComputationPtr labels)
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
  return static_cast<float>(_correct.load(std::memory_order_relaxed)) /
         _num_samples.load(std::memory_order_relaxed);
}

float CategoricalAccuracy::worst() const { return 0.0; }

bool CategoricalAccuracy::betterThan(float a, float b) const { return a > b; }

}  // namespace thirdai::bolt::train::metrics