#include "ModelPipeline.h"
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace thirdai::automl::models {

void ModelPipeline::trainOnFile(
    const std::string& filename, bolt::TrainConfig& train_config,
    std::optional<uint32_t> batch_size_opt,
    const std::optional<ValidationOptions>& validation,
    std::optional<uint32_t> max_in_memory_batches) {
  uint32_t batch_size =
      batch_size_opt.value_or(_train_eval_config.defaultBatchSize());
  trainOnDataLoader(dataset::SimpleFileDataLoader::make(filename, batch_size),
                    train_config, validation, max_in_memory_batches);
}

void ModelPipeline::trainOnDataLoader(
    const std::shared_ptr<dataset::DataLoader>& data_source,
    bolt::TrainConfig& train_config,
    const std::optional<ValidationOptions>& validation,
    std::optional<uint32_t> max_in_memory_batches) {
  _dataset_factory->preprocessDataset(data_source, max_in_memory_batches);
  data_source->restart();

  auto dataset = _dataset_factory->getLabeledDatasetLoader(
      data_source, /* training= */ true);

  updateRehashRebuildInTrainConfig(train_config);

  if (max_in_memory_batches) {
    trainOnStream(dataset, train_config, max_in_memory_batches.value());
  } else {
    trainInMemory(dataset, train_config, validation);
  }
}

py::object ModelPipeline::evaluateOnFile(
    const std::string& filename,
    std::optional<bolt::EvalConfig>& eval_config_opt) {
  return evaluateOnDataLoader(dataset::SimpleFileDataLoader::make(
                                  filename, DEFAULT_EVALUATE_BATCH_SIZE),
                              eval_config_opt);
}

py::object ModelPipeline::evaluateOnDataLoader(
    const std::shared_ptr<dataset::DataLoader>& data_source,
    std::optional<bolt::EvalConfig>& eval_config_opt) {
  auto dataset = _dataset_factory->getLabeledDatasetLoader(
      data_source, /* training= */ false);

  auto [data, labels] =
      dataset->loadInMemory(std::numeric_limits<uint32_t>::max()).value();

  bolt::EvalConfig eval_config =
      eval_config_opt.value_or(bolt::EvalConfig::makeConfig());

  eval_config.returnActivations();

  auto [_, output] = _model->evaluate({data}, labels, eval_config);

  if (auto threshold = _train_eval_config.predictionThreshold()) {
    uint32_t output_dim = output.numNonzerosInOutput();
    for (uint32_t i = 0; i < output.numSamples(); i++) {
      float* activations = output.activationsForSample(i);
      uint32_t prediction_index = argmax(activations, output_dim);

      if (activations[prediction_index] < threshold.value()) {
        activations[prediction_index] = threshold.value() + 0.0001;
      }
    }
  }

  output = _dataset_factory->processEvaluateOutput(output);
  return convertInferenceTrackerToNumpy(output);
}

template py::object ModelPipeline::predict(const LineInput&, bool);
template py::object ModelPipeline::predict(const MapInput&, bool);

template <typename InputType>
py::object ModelPipeline::predict(const InputType& sample,
                                  bool use_sparse_inference) {
  std::vector<BoltVector> inputs = _dataset_factory->featurizeInput(sample);

  BoltVector output =
      _model->predictSingle(std::move(inputs), use_sparse_inference);

  if (auto threshold = _train_eval_config.predictionThreshold()) {
    uint32_t prediction_index = argmax(output.activations, output.len);
    if (output.activations[prediction_index] < threshold.value()) {
      output.activations[prediction_index] = threshold.value() + 0.0001;
    }
  }

  output = _dataset_factory->processOutputVector(output);

  return convertBoltVectorToNumpy(output);
}

template py::object ModelPipeline::predictBatch(const LineInputBatch&, bool);
template py::object ModelPipeline::predictBatch(const MapInputBatch&, bool);

template <typename InputBatchType>
py::object ModelPipeline::predictBatch(const InputBatchType& samples,
                                       bool use_sparse_inference) {
  std::vector<BoltBatch> input_batches =
      _dataset_factory->featurizeInputBatch(samples);

  BoltBatch outputs = _model->predictSingleBatch(std::move(input_batches),
                                                 use_sparse_inference);

  if (auto threshold = _train_eval_config.predictionThreshold()) {
    for (auto& output : outputs) {
      uint32_t prediction_index = argmax(output.activations, output.len);
      if (output.activations[prediction_index] < threshold.value()) {
        output.activations[prediction_index] = threshold.value() + 0.0001;
      }
    }
  }

  for (auto& vector : outputs) {
    vector = _dataset_factory->processOutputVector(vector);
  }

  return convertBoltBatchToNumpy(outputs);
}

template std::vector<dataset::Explanation> ModelPipeline::explain(
    const LineInput&, std::optional<std::variant<uint32_t, std::string>>);
template std::vector<dataset::Explanation> ModelPipeline::explain(
    const MapInput&, std::optional<std::variant<uint32_t, std::string>>);

template <typename InputType>
std::vector<dataset::Explanation> ModelPipeline::explain(
    const InputType& sample,
    std::optional<std::variant<uint32_t, std::string>> target_class) {
  std::optional<uint32_t> target_neuron;
  if (target_class) {
    target_neuron = _dataset_factory->labelToNeuronId(*target_class);
  }

  auto [gradients_indices, gradients_ratio] = _model->getInputGradientSingle(
      /* input_data= */ {_dataset_factory->featurizeInput(sample)},
      /* explain_prediction_using_highest_activation= */ true,
      /* neuron_to_explain= */ target_neuron);
  return _dataset_factory->explain(gradients_indices, gradients_ratio, sample);
}

void ModelPipeline::trainInMemory(
    data::DatasetLoaderPtr& dataset, bolt::TrainConfig train_config,
    const std::optional<ValidationOptions>& validation) {
  auto [train_data, train_labels] =
      dataset->loadInMemory(std::numeric_limits<uint32_t>::max()).value();

  if (validation) {
    auto validation_dataset = _dataset_factory->getLabeledDatasetLoader(
        dataset::SimpleFileDataLoader::make(validation->filename(),
                                            DEFAULT_EVALUATE_BATCH_SIZE),
        /* training= */ false);

    auto [val_data, val_labels] =
        validation_dataset->loadInMemory(std::numeric_limits<uint32_t>::max())
            .value();

    train_config.withValidation(val_data, val_labels,
                                validation->validationConfig());
  }

  uint32_t epochs = train_config.epochs();

  if (_train_eval_config.freezeHashTables() && epochs > 1) {
    train_config.setEpochs(/* new_epochs=*/1);

    _model->train(train_data, train_labels, train_config);

    _model->freezeHashTables(/* insert_labels_if_not_found= */ true);

    train_config.setEpochs(/* new_epochs= */ epochs - 1);
  }

  _model->train(train_data, train_labels, train_config);
}

// We take in the TrainConfig by value to copy it so we can modify the number
// epochs.
void ModelPipeline::trainOnStream(data::DatasetLoaderPtr& dataset,
                                  bolt::TrainConfig train_config,
                                  uint32_t max_in_memory_batches) {
  uint32_t epochs = train_config.epochs();
  // We want a single epoch in the train config in order to train for a single
  // epoch for each pass over the dataset.
  train_config.setEpochs(/* new_epochs= */ 1);

  if (_train_eval_config.freezeHashTables() && epochs > 1) {
    trainSingleEpochOnStream(dataset, train_config, max_in_memory_batches);
    _model->freezeHashTables(/* insert_labels_if_not_found= */ true);

    --epochs;
  }

  for (uint32_t e = 0; e < epochs; e++) {
    trainSingleEpochOnStream(dataset, train_config, max_in_memory_batches);
  }
}

void ModelPipeline::trainSingleEpochOnStream(
    data::DatasetLoaderPtr& dataset, const bolt::TrainConfig& train_config,
    uint32_t max_in_memory_batches) {
  while (auto datasets = dataset->loadInMemory(max_in_memory_batches)) {
    auto& [data, labels] = datasets.value();

    _model->train({data}, labels, train_config);
  }

  dataset->restart();
}

void ModelPipeline::updateRehashRebuildInTrainConfig(
    bolt::TrainConfig& train_config) {
  if (auto hash_table_rebuild =
          _train_eval_config.rebuildHashTablesInterval()) {
    train_config.withRebuildHashTables(hash_table_rebuild.value());
  }

  if (auto reconstruct_hash_fn =
          _train_eval_config.reconstructHashFunctionsInterval()) {
    train_config.withReconstructHashFunctions(reconstruct_hash_fn.value());
  }
}

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

py::object convertInferenceTrackerToNumpy(
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

py::object convertBoltVectorToNumpy(const BoltVector& vector) {
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

py::object convertBoltBatchToNumpy(const BoltBatch& batch) {
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