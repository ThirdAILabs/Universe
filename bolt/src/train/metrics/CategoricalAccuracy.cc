#include "CategoricalAccuracy.h"

namespace thirdai::bolt::train::metrics {

CategoricalAccuracy::CategoricalAccuracy() : _correct(0), _num_samples(0) {}

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

std::string CategoricalAccuracy::name() const { return "categorical_accuracy"; }

void CategoricalAccuracy::record(const BoltVector& output,
                                 const BoltVector& label) {
  uint32_t prediction = output.getHighestActivationId();

  if (label.findActiveNeuronNoTemplate(prediction).activation > 0) {
    _correct++;
  }
  _num_samples++;
}

}  // namespace thirdai::bolt::train::metrics