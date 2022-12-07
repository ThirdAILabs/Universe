#include "ModelPipeline.h"
#include <bolt/src/metrics/Metric.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <telemetry/src/PrometheusClient.h>
#include <limits>

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
    const dataset::DataLoaderPtr& data_source, bolt::TrainConfig& train_config,
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

  // If the model is for binary classification then at the end of each call to
  // train we check to see if there is a prediction theshold that improves
  // performance on some metric. If multiple metrics are specified it defaults
  // to using the first metric.
  if (auto binary_output = BinaryOutputProcessor::cast(_output_processor)) {
    if (validation && !validation->metrics().empty()) {
      std::optional<float> threshold =
          tuneBinaryClassificationPredictionThreshold(
              /* data_source= */ dataset::SimpleFileDataLoader::make(
                  validation->filename(), DEFAULT_EVALUATE_BATCH_SIZE),
              /* metric_name= */ validation->metrics().at(0),
              /* max_num_batches= */ ALL_BATCHES);

      binary_output->setPredictionTheshold(threshold);
    } else if (!train_config.metrics().empty()) {
      // The number of training batches used is capped at 100 in case there is a
      // large training dataset.
      data_source->restart();
      std::optional<float> threshold =
          tuneBinaryClassificationPredictionThreshold(
              /* data_source= */ data_source,
              /* metric_name= */ train_config.metrics().at(0),
              /* max_num_batches= */ MAX_TRAIN_BATCHES_FOR_THRESHOLD_TUNING);

      binary_output->setPredictionTheshold(threshold);
    }
  }

  std::chrono::duration<double> elapsed_time =
      std::chrono::system_clock::now() - start_time;
  telemetry::client.trackTraining(
      /* training_time_seconds = */ elapsed_time.count());
}

py::object ModelPipeline::evaluateOnFile(
    const std::string& filename,
    std::optional<bolt::EvalConfig>& eval_config_opt,
    bool return_predicted_class, bool return_metrics) {
  return evaluateOnDataLoader(dataset::SimpleFileDataLoader::make(
                                  filename, DEFAULT_EVALUATE_BATCH_SIZE),
                              eval_config_opt, return_predicted_class,
                              return_metrics);
}

py::object ModelPipeline::evaluateOnDataLoader(
    const dataset::DataLoaderPtr& data_source,
    std::optional<bolt::EvalConfig>& eval_config_opt,
    bool return_predicted_class, bool return_metrics) {
  auto start_time = std::chrono::system_clock::now();

  auto dataset = _dataset_factory->getLabeledDatasetLoader(
      data_source, /* training= */ false);

  auto [data, labels] = dataset->loadInMemory(ALL_BATCHES).value();

  bolt::EvalConfig eval_config =
      eval_config_opt.value_or(bolt::EvalConfig::makeConfig());

  eval_config.returnActivations();

  auto [metrics, output] = _model->evaluate({data}, labels, eval_config);

  auto py_output = return_metrics ? py::cast(metrics)
                                  : _output_processor->processOutputTracker(
                                        output, return_predicted_class);

  std::chrono::duration<double> elapsed_time =
      std::chrono::system_clock::now() - start_time;
  telemetry::client.trackEvaluate(
      /* evaluate_time_seconds = */ elapsed_time.count());

  return py_output;
}

template py::object ModelPipeline::predict(const LineInput&, bool, bool);
template py::object ModelPipeline::predict(const MapInput&, bool, bool);

template <typename InputType>
py::object ModelPipeline::predict(const InputType& sample,
                                  bool use_sparse_inference,
                                  bool return_predicted_class) {
  auto start_time = std::chrono::system_clock::now();

  std::vector<BoltVector> inputs = _dataset_factory->featurizeInput(sample);

  BoltVector output =
      _model->predictSingle(std::move(inputs), use_sparse_inference);

  auto py_output =
      _output_processor->processBoltVector(output, return_predicted_class);

  std::chrono::duration<double> elapsed_time =
      std::chrono::system_clock::now() - start_time;
  telemetry::client.trackPrediction(
      /* inference_time_seconds = */ elapsed_time.count());

  return py_output;
}

template py::object ModelPipeline::predictBatch(const LineInputBatch&, bool,
                                                bool);
template py::object ModelPipeline::predictBatch(const MapInputBatch&, bool,
                                                bool);

template <typename InputBatchType>
py::object ModelPipeline::predictBatch(const InputBatchType& samples,
                                       bool use_sparse_inference,
                                       bool return_predicted_class) {
  auto start_time = std::chrono::system_clock::now();

  std::vector<BoltBatch> input_batches =
      _dataset_factory->featurizeInputBatch(samples);

  BoltBatch outputs = _model->predictSingleBatch(std::move(input_batches),
                                                 use_sparse_inference);

  auto py_output =
      _output_processor->processBoltBatch(outputs, return_predicted_class);

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

// We take in the TrainConfig by value to copy it so we can modify the number
// epochs.
void ModelPipeline::trainInMemory(
    data::DatasetLoaderPtr& dataset, bolt::TrainConfig train_config,
    const std::optional<ValidationOptions>& validation) {
  auto [train_data, train_labels] = dataset->loadInMemory(ALL_BATCHES).value();

  if (validation) {
    auto validation_dataset = _dataset_factory->getLabeledDatasetLoader(
        dataset::SimpleFileDataLoader::make(validation->filename(),
                                            DEFAULT_EVALUATE_BATCH_SIZE),
        /* training= */ false);

    auto [val_data, val_labels] =
        validation_dataset->loadInMemory(ALL_BATCHES).value();

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

std::optional<float> ModelPipeline::tuneBinaryClassificationPredictionThreshold(
    const dataset::DataLoaderPtr& data_source, const std::string& metric_name,
    uint32_t max_num_batches) {
  auto dataset = _dataset_factory->getLabeledDatasetLoader(
      data_source, /* training= */ false);

  auto [data, labels] = dataset->loadInMemory(max_num_batches).value();

  auto eval_config =
      bolt::EvalConfig::makeConfig().returnActivations().silence();
  auto [_, activations] = _model->evaluate({data}, labels, eval_config);

  auto metric = bolt::makeMetric(metric_name);

  double best_metric_value = metric->worst();
  std::optional<float> best_threshold = std::nullopt;

  for (uint32_t t_idx = 1; t_idx < NUM_THRESHOLDS_TO_CHECK; t_idx++) {
    float threshold = static_cast<float>(t_idx) / NUM_THRESHOLDS_TO_CHECK;

    uint32_t sample_idx = 0;
    for (const auto& label_batch : *labels) {
      for (const auto& label_vec : label_batch) {
        /**
         * The output bolt vector from activations cannot be passed in directly
         * because it doesn't incorporate the threshold, and metrics like
         * categorical_accuracy cannot use a threshold. To solve this we create
         * a new output vector where the neuron with the largest activation is
         * the same as the neuron that would be choosen as the prediction if we
         * applied the given prediction threshold.
         *
         * For metrics like F1 or categorical accuracy the value of the
         * activation does not matter, only the predicted class so this
         * modification does not affect the metric. Metrics like mean squared
         * error do not really make sense to compute at different thresholds
         * anyway and so we can ignore the effect of this modification on them.
         */
        if (activations.activationsForSample(sample_idx++)[1] >= threshold) {
          metric->record(
              /* output= */ BoltVector::makeDenseVector({0, 1.0}),
              /* labels= */ label_vec);
        } else {
          metric->record(
              /* output= */ BoltVector::makeDenseVector({1.0, 0.0}),
              /* labels= */ label_vec);
        }
      }
    }

    if (metric->betterThan(metric->value(), best_metric_value)) {
      best_metric_value = metric->value();
      metric->reset();
      best_threshold = threshold;
    }
  }

  return best_threshold;
}

}  // namespace thirdai::automl::models