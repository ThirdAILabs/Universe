#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/python_bindings/ConversionUtils.h>
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
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::bolt::python {

template <typename PREDICT_INPUT_TYPE>
class AutoClassifierBase {
 public:
  enum class ReturnMode { NumpyArray, ClassName };

  explicit AutoClassifierBase(BoltGraphPtr model, ReturnMode return_mode)
      : _model(std::move(model)), _return_mode(return_mode) {}

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
    auto dataset = getTrainingDataset(data_source, max_in_memory_batches);

    if (max_in_memory_batches) {
      trainOnStream(dataset, learning_rate, epochs,
                    max_in_memory_batches.value());
      return;
    }

    auto [train_data, train_labels] = dataset->loadInMemory();
    trainInMemory(train_data, train_labels, learning_rate, epochs);
  }

  py::object evaluate(const std::string& filename) {
    return evaluate(std::make_shared<dataset::SimpleFileDataLoader>(
        filename, defaultBatchSize()));
  }

  py::object evaluate(const std::shared_ptr<dataset::DataLoader>& data_source) {
    auto dataset = getTestDataset(data_source);

    auto [data, labels] = dataset->loadInMemory();

    PredictConfig predict_cfg = PredictConfig::makeConfig()
                                    .withMetrics(getEvaluationMetrics())
                                    .returnActivations();
    if (useSparseInference()) {
      predict_cfg.enableSparseInference();
    }

    auto [metrics, output] = _model->predict({data}, {labels}, predict_cfg);

    processPredictionsBeforeReturning(output);

    switch (_return_mode) {
      case ReturnMode::NumpyArray:
        return constructNumpyActivationsArrays(metrics, output);
      case ReturnMode::ClassName:
        return py::make_tuple(py::cast(metrics), getClassNames(output));
      default:
        // This cannot be reached but the compiler complains.
        throw std::invalid_argument("Invalid ReturnMode reached.");
    }
  }

  py::object predict(const PREDICT_INPUT_TYPE& sample) {
    BoltVector input = featurizeInputForInference(sample);

    BoltVector output = _model->predictSingle({input}, useSparseInference());

    processPredictionBeforeReturning(output.active_neurons, output.activations,
                                     output.len);

    switch (_return_mode) {
      case ReturnMode::NumpyArray:
        return constructNumpyVector(output);
      case ReturnMode::ClassName:
        return py::cast(getClassName(output.getHighestActivationId()));
      default:
        // This cannot be reached but the compiler complains.
        throw std::invalid_argument("Invalid ReturnMode reached.");
    }
  }

  virtual ~AutoClassifierBase() = default;

 protected:
  /**
   * Constructs a training dataset from the given data loader.
   */
  virtual std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  getTrainingDataset(std::shared_ptr<dataset::DataLoader> data_loader,
                     std::optional<uint64_t> max_in_memory_batches) = 0;

  /**
   * Constructs a test dataset from the given data loader. This is separate from
   * getTrainingDataset because some classifiers like the tabular classifier may
   * need to process training and test datasets seperately.
   */
  virtual std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  getTestDataset(std::shared_ptr<dataset::DataLoader> data_loader) = 0;

  /**
   * Allows for an auto classifier to preprocess the logits of a prediction
   * before returning. For instance in the MultiLabelTextClassifier this is used
   * to apply the threshold so that at least one neuron has a score above the
   * prediction threshold.
   */
  virtual void processPredictionBeforeReturning(uint32_t* active_neurons,
                                                float* activations,
                                                uint32_t len) {
    (void)active_neurons;
    (void)activations;
    (void)len;
  }

  /**
   * This function consumes some inference input type and returns a bolt vector.
   * This is used for the inference api where the model would take in a string
   * or a list of integers, etc. and generate a prediction.
   */
  virtual BoltVector featurizeInputForInference(
      const PREDICT_INPUT_TYPE& input) = 0;

  /**
   * Returns the name of the class associated with the neuron ID. This is
   * primarily for classifiers such as Text and Tabular which take in the labels
   * as strings and assign them to varying neurons.
   */
  virtual std::string getClassName(uint32_t neuron_id) = 0;

  /**
   * Returns the batch size to use if it is not specified by the user.
   */
  virtual uint32_t defaultBatchSize() const = 0;

  /**
   * Determines if the classifier will freeze hash tables after the first epoch.
   */
  virtual bool freezeHashTablesAfterFirstEpoch() const = 0;

  /**
   * Determines if the model will use sparse inference.
   */
  virtual bool useSparseInference() const = 0;

  /**
   * Species any metrics to use during evaluate.
   */
  virtual std::vector<std::string> getEvaluationMetrics() const = 0;

  BoltGraphPtr _model;
  ReturnMode _return_mode;

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

 private:
  void trainInMemory(dataset::BoltDatasetPtr& train_data,
                     dataset::BoltDatasetPtr& train_labels, float learning_rate,
                     uint32_t epochs) {
    if (freezeHashTablesAfterFirstEpoch() && epochs > 1) {
      TrainConfig train_cfg_initial = TrainConfig::makeConfig(learning_rate, 1);
      _model->train({train_data}, train_labels, train_cfg_initial);

      _model->freezeHashTables(/* insert_labels_if_not_found= */ true);

      --epochs;
    }

    TrainConfig train_cfg = TrainConfig::makeConfig(learning_rate, epochs);
    _model->train({train_data}, {train_labels}, train_cfg);
  }

  void trainOnStream(
      std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>& dataset,
      float learning_rate, uint32_t epochs, uint32_t max_in_memory_batches) {
    if (freezeHashTablesAfterFirstEpoch() && epochs > 1) {
      trainSingleEpochOnStream(dataset, learning_rate, max_in_memory_batches);
      _model->freezeHashTables(/* insert_labels_if_not_found= */ true);

      --epochs;
    }

    for (uint32_t e = 0; e < epochs; e++) {
      trainSingleEpochOnStream(dataset, learning_rate, max_in_memory_batches);
    }
  }

  void trainSingleEpochOnStream(
      std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>& dataset,
      float learning_rate, uint32_t max_in_memory_batches) {
    TrainConfig train_config =
        TrainConfig::makeConfig(learning_rate, /* epochs= */ 1);

    while (1) {
      auto [data, labels] = dataset->loadInMemory(max_in_memory_batches);

      _model->train({data}, labels, train_config);

      if (data->numBatches() < max_in_memory_batches) {
        break;
      }
    }

    dataset->restart();
  }

  // TODO(Someone): Allow this to return top-k class names as well.
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

  // Private constructor for cereal.
  AutoClassifierBase() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_model, _return_mode);
  }

  static py::object constructNumpyActivationsArrays(
      InferenceMetricData& metrics, InferenceOutputTracker& output) {
    uint32_t num_samples = output.numSamples();
    uint32_t inference_dim = output.numNonzerosInOutput();

    const uint32_t* active_neurons_ptr =
        output.getNonowningActiveNeuronPointer();
    const float* activations_ptr = output.getNonowningActivationPointer();

    py::object output_handle = py::cast(std::move(output));

    return constructPythonInferenceTuple(
        py::cast(metrics), /* num_samples= */ num_samples,
        /* inference_dim= */ inference_dim, /* activations= */ activations_ptr,
        /* active_neurons= */ active_neurons_ptr,
        /* activation_handle= */ output_handle,
        /* active_neuron_handle= */ output_handle);
  }

  void processPredictionsBeforeReturning(InferenceOutputTracker& output) {
    uint32_t output_dim = output.numNonzerosInOutput();

    for (uint32_t i = 0; i < output.numSamples(); i++) {
      uint32_t* active_neurons = output.activeNeuronsForSample(i);
      float* activations = output.activationsForSample(i);

      processPredictionBeforeReturning(active_neurons, activations, output_dim);
    }
  }

  static py::object constructNumpyVector(BoltVector& output) {
    py::array_t<float, py::array::c_style | py::array::forcecast>
        activations_array(output.len);
    std::copy(output.activations, output.activations + output.len,
              activations_array.mutable_data());

    if (output.isDense()) {
      // This is not a move on return because we are constructing a py::object.
      return std::move(activations_array);
    }

    py::array_t<uint32_t, py::array::c_style | py::array::forcecast>
        active_neurons_array(output.len);
    std::copy(output.active_neurons, output.active_neurons + output.len,
              active_neurons_array.mutable_data());

    return py::make_tuple(active_neurons_array, activations_array);
  }
};

}  // namespace thirdai::bolt::python