#include "Classifier.h"
#include <cereal/archives/binary.hpp>
#include "Train.h"
#include <auto_ml/src/udt/Defaults.h>
#include <pybind11/stl.h>

namespace thirdai::automl::udt::utils {

void thirdai::automl::udt::utils::Classifier::train(
    dataset::DatasetLoaderPtr& dataset, float learning_rate, uint32_t epochs,
    const std::optional<ValidationDatasetLoader>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval,
    licensing::TrainPermissionsToken token) {
  uint32_t batch_size = batch_size_opt.value_or(defaults::BATCH_SIZE);

  bolt::TrainConfig train_config =
      getTrainConfig(epochs, learning_rate, validation, metrics, callbacks,
                     verbose, logging_interval);

  utils::train(_model, dataset, train_config, batch_size, max_in_memory_batches,
               /* freeze_hash_tables= */ _freeze_hash_tables, token);

  /**
   * For binary classification we tune the prediction threshold to optimize
   * some metric. This can improve performance particularly on datasets with a
   * class imbalance.
   */
  if (_model->outputDim() == 2) {
    if (validation && !validation->second.metrics().empty()) {
      validation->first->restart();
      _binary_prediction_threshold =
          tuneBinaryClassificationPredictionThreshold(
              /* dataset= */ validation->first,
              /* metric_name= */ validation->second.metrics().at(0));

    } else if (!train_config.metrics().empty()) {
      dataset->restart();
      _binary_prediction_threshold =
          tuneBinaryClassificationPredictionThreshold(
              /* dataset= */ dataset,
              /* metric_name= */ train_config.metrics().at(0));
    }
  }
}

py::object Classifier::evaluate(dataset::DatasetLoaderPtr& dataset,
                                const std::vector<std::string>& metrics,
                                bool sparse_inference,
                                bool return_predicted_class, bool verbose,
                                bool return_metrics) {
  bolt::EvalConfig eval_config =
      utils::getEvalConfig(metrics, sparse_inference, verbose);

  auto [test_data, test_labels] =
      dataset->loadAll(/* batch_size= */ defaults::BATCH_SIZE, verbose);

  auto [output_metrics, output] =
      _model->evaluate(test_data, test_labels, eval_config);
  if (return_metrics) {
    return py::cast(output_metrics);
  }

  if (return_predicted_class) {
    return predictedClasses(output);
  }

  return utils::convertInferenceTrackerToNumpy(output);
}

py::object Classifier::predict(std::vector<BoltVector>&& inputs,
                               bool sparse_inference,
                               bool return_predicted_class) {
  BoltVector output =
      _model->predictSingle(std::move(inputs), sparse_inference);

  if (return_predicted_class) {
    return py::cast(predictedClass(output));
  }

  return utils::convertBoltVectorToNumpy(output);
}

py::object Classifier::predictBatch(std::vector<BoltBatch>&& batches,
                                    bool sparse_inference,
                                    bool return_predicted_class) {
  BoltBatch outputs =
      _model->predictSingleBatch(std::move(batches), sparse_inference);

  if (return_predicted_class) {
    return predictedClasses(outputs);
  }

  return utils::convertBoltBatchToNumpy(outputs);
}

uint32_t Classifier::predictedClass(const BoltVector& activation_vec) {
  if (_binary_prediction_threshold) {
    return activation_vec.activations[1] >= *_binary_prediction_threshold;
  }
  return activation_vec.getHighestActivationId();
}

py::object Classifier::predictedClasses(bolt::InferenceOutputTracker& output) {
  utils::NumpyArray<uint32_t> predictions(output.numSamples());
  for (uint32_t i = 0; i < output.numSamples(); i++) {
    BoltVector activation_vec = output.getSampleAsNonOwningBoltVector(i);
    predictions.mutable_at(i) = predictedClass(activation_vec);
  }
  return py::object(std::move(predictions));
}

py::object Classifier::predictedClasses(const BoltBatch& outputs) {
  utils::NumpyArray<uint32_t> predictions(outputs.getBatchSize());
  for (uint32_t i = 0; i < outputs.getBatchSize(); i++) {
    predictions.mutable_at(i) = predictedClass(outputs[i]);
  }
  return py::object(std::move(predictions));
}

std::optional<float> Classifier::tuneBinaryClassificationPredictionThreshold(
    const dataset::DatasetLoaderPtr& dataset, const std::string& metric_name) {
  // The number of samples used is capped to ensure tuning is fast even for
  // larger datasets.
  uint32_t num_batches =
      defaults::MAX_SAMPLES_FOR_THRESHOLD_TUNING / defaults::BATCH_SIZE;

  auto loaded_data_opt =
      dataset->loadSome(/* batch_size = */ defaults::BATCH_SIZE, num_batches,
                        /* verbose = */ false);
  if (!loaded_data_opt.has_value()) {
    throw std::invalid_argument("No data found for training.");
  }
  auto loaded_data = *loaded_data_opt;

  auto data = std::move(loaded_data.first);
  auto labels = std::move(loaded_data.second);

  auto eval_config =
      bolt::EvalConfig::makeConfig().returnActivations().silence();
  auto output = _model->evaluate({data}, labels, eval_config);
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

template void Classifier::serialize(cereal::BinaryInputArchive&);
template void Classifier::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Classifier::serialize(Archive& archive) {
  archive(_model, _freeze_hash_tables, _binary_prediction_threshold);
}

}  // namespace thirdai::automl::udt::utils