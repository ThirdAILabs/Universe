#include "ModelPipeline.h"
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <telemetry/src/PrometheusClient.h>

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
  auto start_time = std::chrono::system_clock::now();

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

  std::chrono::duration<double> elapsed_time =
      std::chrono::system_clock::now() - start_time;
  telemetry::client.trackTraining(
      /* training_time_seconds = */ elapsed_time.count());
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
  auto start_time = std::chrono::system_clock::now();

  auto dataset = _dataset_factory->getLabeledDatasetLoader(
      data_source, /* training= */ false);

  auto [data, labels] =
      dataset->loadInMemory(std::numeric_limits<uint32_t>::max()).value();

  bolt::EvalConfig eval_config =
      eval_config_opt.value_or(bolt::EvalConfig::makeConfig());

  eval_config.returnActivations();

  auto [_, output] = _model->evaluate({data}, labels, eval_config);

  auto py_output = _output_processor->processOutputTracker(output);

  std::chrono::duration<double> elapsed_time =
      std::chrono::system_clock::now() - start_time;
  telemetry::client.trackEvaluate(
      /* evaluate_time_seconds = */ elapsed_time.count());

  return py_output;
}

template py::object ModelPipeline::predict(const LineInput&, bool);
template py::object ModelPipeline::predict(const MapInput&, bool);

template <typename InputType>
py::object ModelPipeline::predict(const InputType& sample,
                                  bool use_sparse_inference) {
  auto start_time = std::chrono::system_clock::now();

  std::vector<BoltVector> inputs = _dataset_factory->featurizeInput(sample);

  BoltVector output =
      _model->predictSingle(std::move(inputs), use_sparse_inference);

  auto py_output = _output_processor->processBoltVector(output);

  std::chrono::duration<double> elapsed_time =
      std::chrono::system_clock::now() - start_time;
  telemetry::client.trackPrediction(
      /* inference_time_seconds = */ elapsed_time.count());

  return py_output;
}

template py::object ModelPipeline::predictBatch(const LineInputBatch&, bool);
template py::object ModelPipeline::predictBatch(const MapInputBatch&, bool);

template <typename InputBatchType>
py::object ModelPipeline::predictBatch(const InputBatchType& samples,
                                       bool use_sparse_inference) {
  auto start_time = std::chrono::system_clock::now();

  std::vector<BoltBatch> input_batches =
      _dataset_factory->featurizeInputBatch(samples);

  BoltBatch outputs = _model->predictSingleBatch(std::move(input_batches),
                                                 use_sparse_inference);

  auto py_output = _output_processor->processBoltBatch(outputs);

  std::chrono::duration<double> elapsed_time =
      std::chrono::system_clock::now() - start_time;
  telemetry::client.trackBatchPredictions(
      /* inference_time_seconds = */ elapsed_time.count(),
      /* num_inferences = */ outputs.getBatchSize());

  return py_output;
}

template std::vector<dataset::Explanation> ModelPipeline::explain(
    const LineInput&, std::optional<std::variant<uint32_t, std::string>>);
template std::vector<dataset::Explanation> ModelPipeline::explain(
    const MapInput&, std::optional<std::variant<uint32_t, std::string>>);

template <typename InputType>
std::vector<dataset::Explanation> ModelPipeline::explain(
    const InputType& sample,
    std::optional<std::variant<uint32_t, std::string>> target_class) {
  auto start_time = std::chrono::system_clock::now();

  std::optional<uint32_t> target_neuron;
  if (target_class) {
    target_neuron = _dataset_factory->labelToNeuronId(*target_class);
  }

  auto [gradients_indices, gradients_ratio] = _model->getInputGradientSingle(
      /* input_data= */ {_dataset_factory->featurizeInput(sample)},
      /* explain_prediction_using_highest_activation= */ true,
      /* neuron_to_explain= */ target_neuron);
  auto explanation =
      _dataset_factory->explain(gradients_indices, gradients_ratio, sample);

  std::chrono::duration<double> elapsed_time =
      std::chrono::system_clock::now() - start_time;
  telemetry::client.trackExplanation(
      /* explain_time_seconds = */ elapsed_time.count());

  return explanation;
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

}  // namespace thirdai::automl::models