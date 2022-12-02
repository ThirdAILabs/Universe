#include "InferenceOutputTracker.h"

namespace thirdai::bolt {

InferenceOutputTracker::InferenceOutputTracker(const NodePtr& output_node,
                                               bool save_activations,
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

InferenceOutputTracker::InferenceOutputTracker(
    std::optional<std::vector<uint32_t>> active_neurons,
    std::vector<float> activations, uint32_t num_nonzeros_per_sample)
    : _num_nonzeros_per_sample(num_nonzeros_per_sample),
      _num_samples(activations.size() / num_nonzeros_per_sample),
      _current_vec_index(activations.size() / num_nonzeros_per_sample),
      _activations(std::move(activations)),
      _active_neurons(std::move(active_neurons)) {}

void InferenceOutputTracker::saveOutputBatch(const NodePtr& output_node,
                                             uint32_t batch_size) {
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

const float* InferenceOutputTracker::getNonowningActivationPointer() const {
  if (!_activations.has_value()) {
    return nullptr;
  }
  return _activations->data();
}

const uint32_t* InferenceOutputTracker::getNonowningActiveNeuronPointer()
    const {
  if (!_active_neurons.has_value()) {
    return nullptr;
  }
  return _active_neurons->data();
}

uint32_t* InferenceOutputTracker::activeNeuronsForSample(uint32_t index) {
  if (!_active_neurons.has_value()) {
    return nullptr;
  }
  return _active_neurons->data() + index * _num_nonzeros_per_sample;
}

float* InferenceOutputTracker::activationsForSample(uint32_t index) {
  if (!_activations.has_value()) {
    return nullptr;
  }
  return _activations->data() + index * _num_nonzeros_per_sample;
}

}  // namespace thirdai::bolt