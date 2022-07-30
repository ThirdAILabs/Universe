#pragma once

#include "ExecutionConfig.h"
#include "Node.h"

namespace thirdai::bolt {

// This class is NOT thread safe
class InferenceOutputTracker {
 public:
  // Should only be called after the output_node has been prepared for batch
  // processing
  InferenceOutputTracker(const NodePtr& output_node, bool save_activations,
                         uint32_t total_num_samples)
      : _num_nonzeros_per_sample(output_node->getOutputVector(0).len),
        _num_samples(total_num_samples),
        _current_vec_index(0) {
    // So the linter won't complain in Release mode
    (void)_num_samples;

    bool output_sparse = !output_node->getOutputVector(0).isDense();
    bool save_active_neurons = save_activations && output_sparse;
    uint64_t total_output_length = _num_nonzeros_per_sample * _num_samples;

    try {
      if (save_activations) {
        _activations = std::vector<float>(total_output_length);
      } else {
        _activations = std::nullopt;
      }
      if (save_active_neurons) {
        _active_neurons = std::vector<uint32_t>(total_output_length);
      } else {
        _active_neurons = std::nullopt;
      }
    } catch (std::bad_alloc& e) {
      throw std::invalid_argument(
          "Cannot allocate enough memory for inference output. Split the "
          "dataset into smaller batches and perform inference on each one, or "
          "change the PredictionConfig to do inference without getting the "
          "result.");
    }
  }

  void saveOutputBatch(const NodePtr& output_node, uint32_t batch_size) {
    for (uint32_t vec_id_in_batch = 0; vec_id_in_batch < batch_size;
         vec_id_in_batch++) {
      const auto& current_output_vec =
          output_node->getOutputVector(vec_id_in_batch);
      assert(current_output_vec.len == _num_nonzeros_per_sample);
      assert(_current_vec_index < _num_samples);

      if (activationsSaved()) {
        std::copy(
            current_output_vec.activations,
            current_output_vec.activations + _num_nonzeros_per_sample,
            &(_activations->at(_num_nonzeros_per_sample * _current_vec_index)));
      }

      if (activeNeuronsSaved()) {
        assert(current_output_vec.active_neurons != nullptr);
        std::copy(current_output_vec.active_neurons,
                  current_output_vec.active_neurons + _num_nonzeros_per_sample,
                  &(_active_neurons->at(_num_nonzeros_per_sample *
                                        _current_vec_index)));
      }

      _current_vec_index++;
    }
  }

  // Returns a (possibly null) pointer to the saved activation data.
  // The pointer will be null if we did not save activations.
  const float* getNonowningActivationPointer() const {
    if (!_activations.has_value()) {
      return nullptr;
    }
    return _activations->data();
  }

  // Returns a (possibly null) pointer to the saved active neuron data.
  // The pointer will be null if we did not save activations or if the ouput
  // was dense.
  const uint32_t* getNonowningActiveNeuronPointer() const {
    if (!_active_neurons.has_value()) {
      return nullptr;
    }
    return _active_neurons->data();
  }

  const std::optional<std::vector<float>>& getActivations() const {
    return _activations;
  }

  const std::optional<std::vector<uint32_t>>& getActiveNeurons() const {
    return _active_neurons;
  }

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