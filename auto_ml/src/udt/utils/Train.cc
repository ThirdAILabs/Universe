#include "Train.h"
#include <auto_ml/src/udt/Defaults.h>
#include <dataset/src/Datasets.h>
#include <pybind11/stl.h>

namespace thirdai::automl::udt::utils {

namespace {

void trainSingleEpochOnStream(bolt::BoltGraphPtr& model,
                              dataset::DatasetLoaderPtr& dataset_loader,
                              const bolt::TrainConfig& train_config,
                              uint32_t max_in_memory_batches, size_t batch_size,
                              licensing::TrainPermissionsToken token) {
  while (auto datasets = dataset_loader->loadSome(
             batch_size, /* num_batches = */ max_in_memory_batches,
             /* verbose = */ train_config.verbose())) {
    auto labels = datasets->back();
    datasets->pop_back();

    model->train(datasets.value(), labels, train_config, token);
  }

  dataset_loader->restart();
}

void trainOnStream(bolt::BoltGraphPtr& model,
                   dataset::DatasetLoaderPtr& dataset_loader,
                   bolt::TrainConfig train_config, size_t batch_size,
                   size_t max_in_memory_batches, bool freeze_hash_tables,
                   licensing::TrainPermissionsToken token) {
  uint32_t epochs = train_config.epochs();
  // We want a single epoch in the train config in order to train for a single
  // epoch for each pass over the dataset.
  train_config.setEpochs(/* new_epochs= */ 1);

  if (freeze_hash_tables && epochs > 1) {
    trainSingleEpochOnStream(model, dataset_loader, train_config,
                             max_in_memory_batches, batch_size, token);
    model->freezeHashTables(/* insert_labels_if_not_found= */ true);

    --epochs;
  }

  for (uint32_t e = 0; e < epochs; e++) {
    trainSingleEpochOnStream(model, dataset_loader, train_config,
                             max_in_memory_batches, batch_size, token);
  }
}

}  // namespace

void trainInMemory(bolt::BoltGraphPtr& model,
                   std::vector<dataset::BoltDatasetPtr> datasets,
                   bolt::TrainConfig train_config, bool freeze_hash_tables,
                   licensing::TrainPermissionsToken token) {
  auto labels = datasets.back();
  datasets.pop_back();

  uint32_t epochs = train_config.epochs();

  if (freeze_hash_tables && epochs > 1) {
    train_config.setEpochs(/* new_epochs=*/1);

    model->train(datasets, labels, train_config);

    model->freezeHashTables(/* insert_labels_if_not_found= */ true);

    train_config.setEpochs(/* new_epochs= */ epochs - 1);
  }

  model->train(datasets, labels, train_config, token);
}

void train(bolt::BoltGraphPtr& model, dataset::DatasetLoaderPtr& dataset_loader,
           const bolt::TrainConfig& train_config, size_t batch_size,
           std::optional<size_t> max_in_memory_batches, bool freeze_hash_tables,
           licensing::TrainPermissionsToken token) {
  if (max_in_memory_batches) {
    trainOnStream(model, dataset_loader, train_config, batch_size,
                  max_in_memory_batches.value(), freeze_hash_tables, token);
  } else {
    auto loaded_data = dataset_loader->loadAll(
        /* batch_size = */ batch_size, /* verbose = */ train_config.verbose());
    trainInMemory(model, loaded_data, train_config, freeze_hash_tables, token);
  }
}

bolt::TrainConfig getTrainConfig(
    uint32_t epochs, float learning_rate,
    const std::optional<Validation>& validation,
    const std::vector<std::string>& train_metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval,
    const DataSourceToDatasetLoader& func) {
  bolt::TrainConfig train_config =
      bolt::TrainConfig::makeConfig(learning_rate, epochs)
          .withMetrics(train_metrics)
          .withCallbacks(callbacks);
  if (logging_interval) {
    train_config.withLogLossFrequency(*logging_interval);
  }
  if (!verbose) {
    train_config.silence();
  }
  return train_config;

  if (validation) {
    auto val_data =
        func(validation->data(), /* shuffle = */ false)
            ->loadAll(/* batch_size= */ defaults::BATCH_SIZE, verbose);
    auto val_labels = val_data.back();
    val_data.pop_back();

    bolt::EvalConfig val_config = getEvalConfig(
        validation->metrics(), validation->sparseInference(), verbose);

    train_config.withValidation(val_data, val_labels, val_config,
                                /* validation_frequency = */
                                validation->stepsPerValidation().value_or(0));
  }

  return train_config;
}

bolt::EvalConfig getEvalConfig(const std::vector<std::string>& metrics,
                               bool sparse_inference, bool verbose,
                               bool validation) {
  bolt::EvalConfig eval_config =
      bolt::EvalConfig::makeConfig().withMetrics(metrics);
  if (sparse_inference) {
    eval_config.enableSparseInference();
  }
  if (!verbose) {
    eval_config.silence();
  }
  if (!validation) {
    eval_config.returnActivations();
  }

  return eval_config;
}

uint32_t predictedClass(const BoltVector& activation_vec,
                        std::optional<float> binary_threshold) {
  if (!binary_threshold.has_value()) {
    return activation_vec.getHighestActivationId();
  }
  return activation_vec.activations[1] >= *binary_threshold;
}

py::object predictedClasses(bolt::InferenceOutputTracker& output,
                            std::optional<float> binary_threshold) {
  utils::NumpyArray<uint32_t> predictions(output.numSamples());
  for (uint32_t i = 0; i < output.numSamples(); i++) {
    BoltVector activation_vec = output.getSampleAsNonOwningBoltVector(i);
    predictions.mutable_at(i) =
        predictedClass(activation_vec, binary_threshold);
  }
  return py::object(std::move(predictions));
}

py::object predictedClasses(const BoltBatch& outputs,
                            std::optional<float> binary_threshold) {
  utils::NumpyArray<uint32_t> predictions(outputs.getBatchSize());
  for (uint32_t i = 0; i < outputs.getBatchSize(); i++) {
    predictions.mutable_at(i) = predictedClass(outputs[i], binary_threshold);
  }
  return py::object(std::move(predictions));
}

/**
 * Computes the optimal binary prediction threshold to maximize the given
 * metric on max_num_batches batches of the given dataset. Note: does not
 * shuffle the data to obtain the batches.
 */
std::optional<float> tuneBinaryClassificationPredictionThreshold(
    const dataset::DataSourcePtr& data_source, const std::string& metric_name,
    size_t batch_size, const bolt::BoltGraphPtr& model,
    const DataSourceToDatasetLoader& func) {
  // The number of samples used is capped to ensure tuning is fast even for
  // larger datasets.
  uint32_t num_batches =
      defaults::MAX_SAMPLES_FOR_THRESHOLD_TUNING / batch_size;

  auto dataset = func(data_source, /* shuffle = */ false);

  auto loaded_data_opt =
      dataset->loadSome(/* batch_size = */ defaults::BATCH_SIZE, num_batches,
                        /* verbose = */ false);
  if (!loaded_data_opt.has_value()) {
    throw std::invalid_argument("No data found for training.");
  }
  auto loaded_data = *loaded_data_opt;

  auto labels = loaded_data.back();
  loaded_data.pop_back();

  auto eval_config =
      bolt::EvalConfig::makeConfig().returnActivations().silence();
  auto output = model->evaluate(loaded_data, labels, eval_config);
  auto& activations = output.second;

  double best_metric_value = bolt::makeMetric(metric_name)->worst();
  std::optional<float> best_threshold = std::nullopt;

#pragma omp parallel for default(none) shared( \
    labels, best_metric_value, best_threshold, metric_name, activations)
  for (uint32_t t_idx = 1; t_idx < defaults::NUM_THRESHOLDS_TO_CHECK; t_idx++) {
    auto metric = bolt::makeMetric(metric_name);

    float threshold =
        static_cast<float>(t_idx) / defaults::NUM_THRESHOLDS_TO_CHECK;

    uint32_t sample_idx = 0;
    for (const auto& label_batch : *labels) {
      for (const auto& label_vec : label_batch) {
        /**
         * The output bolt vector from activations cannot be passed in
         * directly because it doesn't incorporate the threshold, and
         * metrics like categorical_accuracy cannot use a threshold. To
         * solve this we create a new output vector where the neuron with
         * the largest activation is the same as the neuron that would be
         * choosen as the prediction if we applied the given prediction
         * threshold.
         *
         * For metrics like F1 or categorical accuracy the value of the
         * activation does not matter, only the predicted class so this
         * modification does not affect the metric. Metrics like mean
         * squared error do not really make sense to compute at different
         * thresholds anyway and so we can ignore the effect of this
         * modification on them.
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

#pragma omp critical
    if (metric->betterThan(metric->value(), best_metric_value)) {
      best_metric_value = metric->value();
      best_threshold = threshold;
    }
  }

  return best_threshold;
}

std::optional<float> getBinaryClassificationPredictionThreshold(
    const dataset::DataSourcePtr& data,
    const std::optional<Validation>& validation, size_t& batch_size,
    bolt::TrainConfig& train_config, const bolt::BoltGraphPtr& model,
    const DataSourceToDatasetLoader& func) {
  if (model->outputDim() == 2) {
    if (validation && !validation->metrics().empty()) {
      validation->data()->restart();
      return tuneBinaryClassificationPredictionThreshold(
          /* data_source= */ validation->data(),
          /* metric_name= */ validation->metrics().at(0), batch_size, model,
          func);
    }
    if (!train_config.metrics().empty()) {
      data->restart();
      return tuneBinaryClassificationPredictionThreshold(
          /* data_source= */ data,
          /* metric_name= */ train_config.metrics().at(0), batch_size, model,
          func);
    }
  }
  return std::nullopt;
}

py::object evaluate(const std::vector<std::string>& metrics,
                    bool sparse_inference, bool return_predicted_class,
                    bool verbose, bool return_metrics,
                    const bolt::BoltGraphPtr& model,
                    const dataset::DatasetLoaderPtr& dataset_loader,
                    const std::optional<float>& binary_prediction_threshold) {
  bolt::EvalConfig eval_config =
      utils::getEvalConfig(metrics, sparse_inference, verbose);

  auto datasets =
      dataset_loader->loadAll(/* batch_size= */ defaults::BATCH_SIZE, verbose);
  auto labels = datasets.back();
  datasets.pop_back();

  auto [output_metrics, output] =
      model->evaluate(datasets, labels, eval_config);
  if (return_metrics) {
    return py::cast(output_metrics);
  }

  if (return_predicted_class) {
    return utils::predictedClasses(output, binary_prediction_threshold);
  }

  return utils::convertInferenceTrackerToNumpy(output);
}

std::optional<float> trainClassifier(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<Validation>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval,
    const utils::DataSourceToDatasetLoader& source_to_loader_func,
    bolt::BoltGraphPtr& model) {
  size_t batch_size = batch_size_opt.value_or(defaults::BATCH_SIZE);

  bolt::TrainConfig train_config = utils::getTrainConfig(
      epochs, learning_rate, validation, metrics, callbacks, verbose,
      logging_interval, source_to_loader_func);

  auto train_dataset_loader = source_to_loader_func(data, /* shuffle = */ true);

  utils::train(model, train_dataset_loader, train_config, batch_size,
               max_in_memory_batches,
               /* freeze_hash_tables= */ false,
               licensing::TrainPermissionsToken(data->resourceName()));

  /**
   * For binary classification we tune the prediction threshold to optimize some
   * metric. This can improve performance particularly on datasets with a class
   * imbalance.
   */
  return utils::getBinaryClassificationPredictionThreshold(
      data, validation, batch_size, train_config, model, source_to_loader_func);
}

}  // namespace thirdai::automl::udt::utils