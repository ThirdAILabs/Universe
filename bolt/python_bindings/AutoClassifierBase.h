#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include "ConversionUtils.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <dataset/src/DataLoader.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/InMemoryDataset.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <limits>
#include <optional>

namespace py = pybind11;

namespace thirdai::bolt::python {

class AutoClassifierBase {
 public:
  enum class ReturnMode { NumpyArray, NumpyArrayWithThresholding, ClassName };

  explicit AutoClassifierBase(BoltGraphPtr model, ReturnMode return_mode,
                              std::optional<float> threshold = std::nullopt)
      : _model(std::move(model)), _return_mode(return_mode) {
    if (_return_mode == ReturnMode::NumpyArrayWithThresholding) {
      _threshold = threshold.value_or(0.95);
    }
  }

  void train(const std::string& filename, uint32_t epochs, float learning_rate,
             std::optional<uint32_t> batch_size = std::nullopt,
             std::optional<uint32_t> max_in_memory_batches = std::nullopt) {
    auto data_source = std::make_shared<dataset::SimpleFileDataLoader>(
        filename, batch_size.value_or(defaultBatchSize()));

    train(data_source, epochs, learning_rate, max_in_memory_batches);
  }

  void train(const std::shared_ptr<dataset::DataLoader>& data_source,
             uint32_t epochs, float learning_rate,
             std::optional<uint32_t> max_in_memory_batches = std::nullopt) {
    auto batch_processor =
        getTrainingBatchProcessor(data_source, max_in_memory_batches);
    data_source->restart();

    dataset::StreamingDataset<BoltBatch, BoltBatch> dataset(data_source,
                                                            batch_processor);

    if (max_in_memory_batches) {
      trainOnStream(dataset, learning_rate, epochs,
                    max_in_memory_batches.value());
    }

    auto [train_data, train_labels] = dataset.loadInMemory();
    trainInMemory(train_data, train_labels, learning_rate, epochs);
  }

  py::object predict(const std::string& filename) {
    return predict(std::make_shared<dataset::SimpleFileDataLoader>(
        filename, defaultBatchSize()));
  }

  py::object predict(const std::shared_ptr<dataset::DataLoader>& data_source) {
    auto batch_processor = getPredictBatchProcessor();

    dataset::StreamingDataset<BoltBatch, BoltBatch> data_loader(
        data_source, batch_processor);

    auto [data, labels] = data_loader.loadInMemory();

    PredictConfig predict_cfg = PredictConfig::makeConfig()
                                    .withMetrics(getPredictMetrics())
                                    .returnActivations();
    if (useSparseInference()) {
      predict_cfg.enableSparseInference();
    }

    auto [metrics, output] = _model->predict({data}, {labels}, predict_cfg);

    if (_return_mode == ReturnMode::NumpyArrayWithThresholding) {
      thresholdActivations(output, _threshold);
    }

    switch (_return_mode) {
      case ReturnMode::NumpyArray:
      case ReturnMode::NumpyArrayWithThresholding:
        return constructNumpyActivationsArrays(metrics, output);
      case ReturnMode::ClassName:
        return py::make_tuple(py::cast(metrics), getClassNames(output));
    }
  }

  py::object predict_single(const py::object& sample) {
    BoltVector input = featurizeInputForInference(sample);

    BoltVector output = _model->predictSingle({input}, useSparseInference());

    if (_return_mode == ReturnMode::NumpyArrayWithThresholding) {
      uint32_t max_id = output.getHighestActivationId();

      if (output.findActiveNeuronNoTemplate(max_id).activation < _threshold) {
        output.activations[output.findActiveNeuronNoTemplate(max_id)
                               .pos.value()] = _threshold + 0.0001;
      }
    }

    switch (_return_mode) {
      case ReturnMode::NumpyArray:
      case ReturnMode::NumpyArrayWithThresholding:
        return constructNumpyVector(output);
      case ReturnMode::ClassName:
        return py::cast(getClassName(output.getHighestActivationId()));
    }
  }

 private:
  void trainInMemory(dataset::BoltDatasetPtr& train_data,
                     dataset::BoltDatasetPtr& train_labels, float learning_rate,
                     uint32_t epochs) {
    if (freezeHashTables() && epochs > 1) {
      TrainConfig train_cfg_initial = TrainConfig::makeConfig(learning_rate, 1);
      _model->train({train_data}, train_labels, train_cfg_initial);

      _model->freezeHashTables(/* insert_labels_if_not_found= */ true);

      --epochs;
    }

    TrainConfig train_cfg = TrainConfig::makeConfig(learning_rate, epochs);
    _model->train({train_data}, {train_labels}, train_cfg);
  }

  void trainOnStream(dataset::StreamingDataset<BoltBatch, BoltBatch>& dataset,
                     float learning_rate, uint32_t epochs,
                     uint32_t max_in_memory_batches) {
    if (freezeHashTables() && epochs > 1) {
      trainSingleEpochOnStream(dataset, learning_rate, max_in_memory_batches);
      _model->freezeHashTables(/* insert_labels_if_not_found= */ true);

      --epochs;
    }

    for (uint32_t e = 0; e < epochs; e++) {
      trainSingleEpochOnStream(dataset, learning_rate, max_in_memory_batches);
    }
  }

  void trainSingleEpochOnStream(
      dataset::StreamingDataset<BoltBatch, BoltBatch>& dataset,
      float learning_rate, uint32_t max_in_memory_batches) {
    while (1) {
      auto [data, labels] = dataset.loadInMemory(max_in_memory_batches);

      if (data->len() == 0) {
        break;
      }

      TrainConfig train_config =
          TrainConfig::makeConfig(learning_rate, /* epochs= */ 1);
      _model->train({data}, labels, train_config);
    }

    dataset.restart();
  }

 protected:
  /**
   * Interface for constructing batch processor and featurizing data.
   */

  virtual dataset::GenericBatchProcessorPtr getTrainingBatchProcessor(
      std::shared_ptr<dataset::DataLoader> data_loader,
      std::optional<uint64_t> max_in_memory_batches) = 0;

  virtual dataset::GenericBatchProcessorPtr getPredictBatchProcessor() = 0;

  virtual BoltVector featurizeInputForInference(const py::object& input) = 0;

  virtual std::string getClassName(uint32_t neuron_id) = 0;

  /**
   * Interface for other options related to training and prediction.
   */

  virtual uint32_t defaultBatchSize() const = 0;

  virtual bool freezeHashTables() const = 0;

  virtual bool useSparseInference() const = 0;

  virtual std::vector<std::string> getPredictMetrics() const = 0;

  BoltGraphPtr _model;
  ReturnMode _return_mode;
  float _threshold;

 private:
  // Private constructor for cereal.
  AutoClassifierBase() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_model, _return_mode, _threshold);
  }

  static py::object constructNumpyActivationsArrays(
      InferenceMetricData& metrics, InferenceOutputTracker& output) {
    uint32_t num_samples = output.numSamples();
    uint32_t inference_dim = output.numNonzerosInOutput();
    py::object output_handle = py::cast(std::move(output));

    const uint32_t* active_neurons_ptr =
        output.getNonowningActiveNeuronPointer();
    const float* activations_ptr = output.getNonowningActivationPointer();

    return constructPythonInferenceTuple(
        py::cast(metrics), /* num_samples= */ num_samples,
        /* inference_dim= */ inference_dim, /* activations= */ activations_ptr,
        /* active_neurons= */ active_neurons_ptr,
        /* activation_handle= */ output_handle,
        /* active_neuron_handle= */ output_handle);
  }

  static void thresholdActivations(InferenceOutputTracker& output,
                                   float threshold) {
    uint32_t output_dim = output.numNonzerosInOutput();

    for (uint32_t i = 0; i < output.numSamples(); i++) {
      float* activations = output.activationsForSample(i);
      uint32_t max_index = getMaxIndex(activations, output_dim);

      if (activations[max_index] < threshold) {
        activations[max_index] = threshold + 0.0001;
      }
    }
  }

  py::list getClassNames(InferenceOutputTracker& output) {
    py::list output_class_names;

    uint32_t output_dim = output.numNonzerosInOutput();

    for (uint32_t i = 0; i < output.numSamples(); i++) {
      uint32_t* active_neurons = output.activeNeuronsForSample(i);
      float* activations = output.activationsForSample(i);

      uint32_t max_index = getMaxIndex(activations, output_dim);

      uint32_t pred =
          active_neurons == nullptr ? max_index : active_neurons[max_index];

      output_class_names.append(getClassName(pred));
    }

    return output_class_names;
  }

  static uint32_t getMaxIndex(const float* const values, uint32_t len) {
    uint32_t max_index = 0;
    float max_value = -std::numeric_limits<float>::max();

    for (uint32_t i = 0; i < len; i++) {
      if (values[i] > max_value) {
        max_value = values[i];
        max_index = i;
      }
    }
    return max_index;
  }

  static py::object constructNumpyVector(BoltVector& output) {
    py::array_t<float, py::array::c_style | py::array::forcecast>
        activations_array(output.len);
    std::copy(output.activations, output.activations + output.len,
              activations_array.data());

    if (output.isDense()) {
      // This is not a move on return because we are constructing a py::object.
      return std::move(activations_array);
    }

    py::array_t<uint32_t, py::array::c_style | py::array::forcecast>
        active_neurons_array(output.len);
    std::copy(output.active_neurons, output.active_neurons + output.len,
              active_neurons_array.data());

    return py::make_tuple(active_neurons_array, activations_array);
  }

 protected:
  static float getHiddenLayerSparsity(uint64_t layer_dim) {
    if (layer_dim < 300) {
      return 1.0;
    }
    if (layer_dim < 1000) {
      return 0.2;
    }
    if (layer_dim < 4000) {
      return 0.1;
    }
    if (layer_dim < 10000) {
      return 0.05;
    }
    if (layer_dim < 30000) {
      return 0.01;
    }
    return 0.005;
  }
};

}  // namespace thirdai::bolt::python