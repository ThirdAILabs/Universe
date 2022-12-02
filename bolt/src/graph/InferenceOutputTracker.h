#pragma once

#include "Node.h"
#include <bolt/src/metrics/MetricAggregator.h>

namespace thirdai::bolt {

// This class is NOT thread safe
class InferenceOutputTracker {
 public:
  // Should only be called after the output_node has been prepared for batch
  // processing
  InferenceOutputTracker(const NodePtr& output_node, bool save_activations,
                         uint32_t total_num_samples);

  void saveOutputBatch(const NodePtr& output_node, uint32_t batch_size);

  // Returns a (possibly null) pointer to the saved activation data.
  // The pointer will be null if we did not save activations.
  const float* getNonowningActivationPointer() const;
  // Returns a (possibly null) pointer to the saved active neuron data.
  // The pointer will be null if we did not save activations or if the ouput
  // was dense.
  const uint32_t* getNonowningActiveNeuronPointer() const;

  uint32_t* activeNeuronsForSample(uint32_t index);

  float* activationsForSample(uint32_t index);

  BoltVector getSampleAsNonOwningBoltVector(uint32_t index);

  uint32_t numNonzerosInOutput() const { return _num_nonzeros_per_sample; }

  uint32_t numSamples() const { return _num_samples; }

  // This will only return a valid value AFTER the object has been constructed,
  // so don't call it from the constructor
  bool activationsSaved() const { return _activations.has_value(); }

  // This will only return a valid value AFTER the object has been constructed,
  // so don't call it from the constructor
  bool activeNeuronsSaved() const { return _active_neurons.has_value(); }

 private:
  uint64_t _num_nonzeros_per_sample;
  uint64_t _num_samples;
  uint64_t _current_vec_index;
  std::optional<std::vector<float>> _activations;
  std::optional<std::vector<uint32_t>> _active_neurons;
};

using InferenceResult = std::pair<InferenceMetricData, InferenceOutputTracker>;

}  // namespace thirdai::bolt