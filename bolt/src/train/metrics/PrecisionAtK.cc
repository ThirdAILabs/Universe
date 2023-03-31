#include "PrecisionAtK.h"
#include <bolt_vector/src/BoltVector.h>
#include <atomic>

namespace thirdai::bolt::train::metrics {

PrecisionAtK::PrecisionAtK(nn::autograd::ComputationPtr outputs,
                                         nn::autograd::ComputationPtr labels,
                                         uint32_t k)
    : _outputs(std::move(outputs)),
      _labels(std::move(labels)),
      _correct(0),
      _num_samples(0),
      _k(k) {}

void PrecisionAtK::record(uint32_t index_in_batch) {
  const BoltVector& output = _outputs->tensor()->getVector(index_in_batch);
  const BoltVector& label = _labels->tensor()->getVector(index_in_batch);

  TopKActivationsQueue topKPredictions = output.findKLargestActivations(_k);

  while (!topKPredictions.empty()) {
      ValueIndexPair valueIndex = topKPredictions.top();
      uint32_t prediction = valueIndex.second;
      if (label.findActiveNeuronNoTemplate(prediction).activation > 0) {
        _correct++;
      }
      topKPredictions.pop();
  }
  _num_samples += _k; 
}

void PrecisionAtK::reset() {
  _correct = 0;
  _num_samples = 0;
}

float PrecisionAtK::value() const {
  // We are using memory order relaxed because we don't need a strict ordering
  // between concurrent accesses, just atomic guarentees.
  return static_cast<float>(_correct.load(std::memory_order_relaxed)) /
         _num_samples.load(std::memory_order_relaxed);
}

float PrecisionAtK::worst() const { return 0.0; }

bool PrecisionAtK::betterThan(float a, float b) const { return a > b; }

}  // namespace thirdai::bolt::train::metrics