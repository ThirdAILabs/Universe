#include "OutputProcessor.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

namespace thirdai::automl::models {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

py::object CategoricalOutputProcessor::processBoltVector(BoltVector& output) {
  if (_prediction_threshold) {
    uint32_t prediction_index = argmax(output.activations, output.len);
    if (output.activations[prediction_index] < _prediction_threshold.value()) {
      output.activations[prediction_index] =
          _prediction_threshold.value() + 0.0001;
    }
  }

  return convertBoltVectorToNumpy(output);
}

py::object CategoricalOutputProcessor::processBoltBatch(BoltBatch& outputs) {
  if (_prediction_threshold) {
    for (auto& output : outputs) {
      uint32_t prediction_index = argmax(output.activations, output.len);
      if (output.activations[prediction_index] <
          _prediction_threshold.value()) {
        output.activations[prediction_index] =
            _prediction_threshold.value() + 0.0001;
      }
    }
  }

  return convertBoltBatchToNumpy(outputs);
}

py::object CategoricalOutputProcessor::processOutputTracker(
    bolt::InferenceOutputTracker& output) {
  if (_prediction_threshold) {
    uint32_t output_dim = output.numNonzerosInOutput();
    for (uint32_t i = 0; i < output.numSamples(); i++) {
      float* activations = output.activationsForSample(i);
      uint32_t prediction_index = argmax(activations, output_dim);

      if (activations[prediction_index] < _prediction_threshold.value()) {
        activations[prediction_index] = _prediction_threshold.value() + 0.0001;
      }
    }
  }

  return convertInferenceTrackerToNumpy(output);
}

py::object RegressionOutputProcessor::processBoltVector(BoltVector& output) {
  float value = _regression_binning.unbinActivations(
      output.active_neurons, output.activations, output.len);

  NumpyArray<float> output_array(1U);
  output_array.mutable_at(0) = value;

  return py::object(std::move(output_array));
}

py::object RegressionOutputProcessor::processBoltBatch(BoltBatch& outputs) {
  NumpyArray<float> output_array(/* shape= */ {outputs.getBatchSize(), 1U});

  for (uint32_t vec_id = 0; vec_id < outputs.getBatchSize(); vec_id++) {
    float value = _regression_binning.unbinActivations(
        outputs[vec_id].active_neurons, outputs[vec_id].activations,
        outputs[vec_id].len);

    output_array.mutable_at(vec_id, 0) = value;
  }

  return py::object(std::move(output_array));
}

py::object RegressionOutputProcessor::processOutputTracker(
    bolt::InferenceOutputTracker& output) {
  NumpyArray<float> output_array(/* shape= */ {output.numSamples(), 1U});

  for (uint32_t i = 0; i < output.numSamples(); i++) {
    output_array.mutable_at(i, 0) = _regression_binning.unbinActivations(
        output.activeNeuronsForSample(i), output.activationsForSample(i),
        output.numNonzerosInOutput());
  }

  return py::object(std::move(output_array));
}

py::object BinaryOutputProcessor::processBoltVector(BoltVector& output) {
  return convertBoltVectorToNumpy(output);
}

py::object BinaryOutputProcessor::processBoltBatch(BoltBatch& outputs) {
  return convertBoltBatchToNumpy(outputs);
}

py::object BinaryOutputProcessor::processOutputTracker(
    bolt::InferenceOutputTracker& output) {
  return convertInferenceTrackerToNumpy(output);
}

py::object OutputProcessor::convertInferenceTrackerToNumpy(
    bolt::InferenceOutputTracker& output) {
  uint32_t num_samples = output.numSamples();
  uint32_t inference_dim = output.numNonzerosInOutput();

  const uint32_t* active_neurons_ptr = output.getNonowningActiveNeuronPointer();
  const float* activations_ptr = output.getNonowningActivationPointer();

  py::object output_handle = py::cast(std::move(output));

  NumpyArray<float> activations_array(
      /* shape= */ {num_samples, inference_dim},
      /* strides= */ {inference_dim * sizeof(float), sizeof(float)},
      /* ptr= */ activations_ptr, /* base= */ output_handle);

  if (!active_neurons_ptr) {
    return py::object(std::move(activations_array));
  }

  // See comment above activations_array for the python memory reasons behind
  // passing in active_neuron_handle
  NumpyArray<uint32_t> active_neurons_array(
      /* shape= */ {num_samples, inference_dim},
      /* strides= */ {inference_dim * sizeof(uint32_t), sizeof(uint32_t)},
      /* ptr= */ active_neurons_ptr, /* base= */ output_handle);

  return py::make_tuple(std::move(activations_array),
                        std::move(active_neurons_array));
}

py::object OutputProcessor::convertBoltVectorToNumpy(const BoltVector& vector) {
  NumpyArray<float> activations_array(vector.len);
  std::copy(vector.activations, vector.activations + vector.len,
            activations_array.mutable_data());

  if (vector.isDense()) {
    return py::object(std::move(activations_array));
  }

  NumpyArray<uint32_t> active_neurons_array(vector.len);
  std::copy(vector.active_neurons, vector.active_neurons + vector.len,
            active_neurons_array.mutable_data());

  return py::make_tuple(active_neurons_array, activations_array);
}

py::object OutputProcessor::convertBoltBatchToNumpy(const BoltBatch& batch) {
  uint32_t length = batch[0].len;

  NumpyArray<float> activations_array(
      /* shape= */ {batch.getBatchSize(), length});

  std::optional<NumpyArray<uint32_t>> active_neurons_array = std::nullopt;
  if (!batch[0].isDense()) {
    active_neurons_array =
        NumpyArray<uint32_t>(/* shape= */ {batch.getBatchSize(), length});
  }

  for (uint32_t i = 0; i < batch.getBatchSize(); i++) {
    if (batch[i].len != length) {
      throw std::invalid_argument(
          "Cannot convert BoltBatch without constant lengths to a numpy "
          "array.");
    }
    if (batch[i].isDense() != !active_neurons_array.has_value()) {
      throw std::invalid_argument(
          "Cannot convert BoltBatch without constant sparsity to a numpy "
          "array.");
    }

    std::copy(batch[i].activations, batch[i].activations + length,
              activations_array.mutable_data() + i * length);
    if (active_neurons_array) {
      std::copy(batch[i].active_neurons, batch[i].active_neurons + length,
                active_neurons_array->mutable_data() + i * length);
    }
  }

  if (active_neurons_array) {
    return py::make_tuple(std::move(active_neurons_array.value()),
                          std::move(activations_array));
  }
  return py::object(std::move(activations_array));
}
}  // namespace thirdai::automl::models

CEREAL_REGISTER_TYPE(thirdai::automl::models::CategoricalOutputProcessor)
CEREAL_REGISTER_TYPE(thirdai::automl::models::RegressionOutputProcessor)