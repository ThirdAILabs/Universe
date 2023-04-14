#include "Classifier.h"
#include <cereal/archives/binary.hpp>
#include <bolt/python_bindings/NumpyConversions.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <pybind11/stl.h>

namespace thirdai::automl::udt::utils {

py::object thirdai::automl::udt::utils::Classifier::train(
    dataset::DatasetLoaderPtr& dataset, float learning_rate, uint32_t epochs,
    const ValidationDatasetLoader& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<bolt::train::callbacks::CallbackPtr>& callbacks,
    bool verbose, std::optional<uint32_t> logging_interval,
    licensing::TrainPermissionsToken token) {
  (void)token;

  uint32_t batch_size = batch_size_opt.value_or(defaults::BATCH_SIZE);

  bolt::train::Trainer trainer(_model);

  bolt::train::metrics::History history;
  if (_freeze_hash_tables) {
    history = trainer.train_with_dataset_loader(
        dataset, learning_rate, /* epochs= */ 1, batch_size,
        max_in_memory_batches, metrics, validation.first,
        validation.second.metrics(), validation.second.stepsPerValidation(),
        validation.second.sparseInference(), callbacks,
        /* autotune_rehash_rebuild= */ true, verbose, logging_interval);

    _model->freezeHashTables(/* insert_labels_if_not_found= */ true);

    dataset->restart();
    epochs--;
  }

  if (epochs > 0) {
    history = trainer.train_with_dataset_loader(
        dataset, learning_rate, epochs, batch_size, max_in_memory_batches,
        metrics, validation.first, validation.second.metrics(),
        validation.second.stepsPerValidation(),
        validation.second.sparseInference(), callbacks,
        /* autotune_rehash_rebuild= */ true, verbose, logging_interval);
  }

  /**
   * For binary classification we tune the prediction threshold to optimize
   * some metric. This can improve performance particularly on datasets with
   * a class imbalance.
   */
  if (_model->outputs().at(0)->dim() == 2) {
    if (!validation.second.metrics().empty()) {
      validation.first->restart();
      _binary_prediction_threshold =
          tuneBinaryClassificationPredictionThreshold(
              /* dataset= */ validation.first,
              /* metric_name= */ validation.second.metrics().at(0));

    } else if (!metrics.empty()) {
      dataset->restart();
      _binary_prediction_threshold =
          tuneBinaryClassificationPredictionThreshold(
              /* dataset= */ dataset,
              /* metric_name= */ metrics.at(0));
    }
  }

  return py::cast(history);
}

py::object Classifier::evaluate(dataset::DatasetLoaderPtr& dataset,
                                const std::vector<std::string>& metrics,
                                bool sparse_inference, bool verbose) {
  bolt::train::Trainer trainer(_model);

  auto history = trainer.validate_with_dataset_loader(
      dataset, metrics, sparse_inference, verbose);

  return py::cast(history);
}

py::object Classifier::predict(const bolt::nn::tensor::TensorList& inputs,
                               bool sparse_inference,
                               bool return_predicted_class) {
  auto output = _model->forward(inputs, sparse_inference).at(0);

  if (return_predicted_class) {
    return predictedClasses(output);
  }

  return bolt::nn::python::tensorToNumpy(output);
}

py::object Classifier::embedding(const bolt::nn::tensor::TensorList& inputs) {
  // TODO(Nicholas): Sparsity could speed this up, and wouldn't affet the
  // embeddings if the sparsity is in the output layer and the embeddings are
  // from the hidden layer.
  _model->forward(inputs, /* use_sparsity= */ false);

  return bolt::nn::python::tensorToNumpy(_emb->tensor());
}

uint32_t Classifier::predictedClass(const BoltVector& output) {
  if (_binary_prediction_threshold) {
    return output.activations[1] >= *_binary_prediction_threshold;
  }
  return output.getHighestActivationId();
}

py::object Classifier::predictedClasses(
    const bolt::nn::tensor::TensorPtr& output) {
  if (output->batchSize() == 1) {
    return py::cast(predictedClass(output->getVector(0)));
  }

  NumpyArray<uint32_t> predictions(output->batchSize());
  for (uint32_t i = 0; i < output->batchSize(); i++) {
    predictions.mutable_at(i) = predictedClass(output->getVector(i));
  }
  return py::object(std::move(predictions));
}

std::vector<std::vector<float>> Classifier::getBinaryClassificationScores(
    const dataset::BoltDatasetList& dataset) {
  auto tensor_batches =
      bolt::train::convertDatasets(dataset, _model->inputDims());

  std::vector<std::vector<float>> scores;
  scores.reserve(tensor_batches.size());

  for (const auto& batch : tensor_batches) {
    auto output = _model->forward(batch, /* use_sparsity= */ false).at(0);

    std::vector<float> batch_scores;
    for (uint32_t i = 0; i < output->batchSize(); i++) {
      batch_scores.push_back(output->getVector(i).activations[1]);
    }
    scores.push_back(std::move(batch_scores));
  }

  return scores;
}

// Splits a vector of datasets as returned by a dataset loader (where the labels
// are the last dataset in the list)
std::pair<dataset::BoltDatasetList, dataset::BoltDatasetPtr> splitDataLabels(
    dataset::BoltDatasetList&& datasets) {
  auto labels = datasets.back();
  datasets.pop_back();
  return {datasets, labels};
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

  // Did this instead of structured binding auto [data, labels] = ...
  // since Clang-Tidy would throw this error around pragma omp parallel:
  // "reference to local binding 'labels' declared in enclosing function"
  auto split_data = splitDataLabels(std::move(*loaded_data_opt));
  auto labels = std::move(split_data.second);

  auto scores = getBinaryClassificationScores(split_data.first);

  double best_metric_value = bolt::makeMetric(metric_name)->worst();
  std::optional<float> best_threshold = std::nullopt;

#pragma omp parallel for default(none) \
    shared(labels, best_metric_value, best_threshold, metric_name, scores)
  for (uint32_t t_idx = 1; t_idx < defaults::NUM_THRESHOLDS_TO_CHECK; t_idx++) {
    // TODO(Nicholas): This is still using the old metric from bolt v1. The bolt
    // v2 metrics are more intertwined with the computation graph and would be
    // harder to use in this way.
    auto metric = bolt::makeMetric(metric_name);

    float threshold =
        static_cast<float>(t_idx) / defaults::NUM_THRESHOLDS_TO_CHECK;

    for (uint32_t batch_idx = 0; batch_idx < scores.size(); batch_idx++) {
      for (uint32_t vec_idx = 0; vec_idx < scores.at(batch_idx).size();
           vec_idx++) {
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
        if (scores.at(batch_idx).at(vec_idx) >= threshold) {
          metric->record(
              /* output= */ BoltVector::makeDenseVector({0, 1.0}),
              /* labels= */ labels->at(batch_idx)[vec_idx]);
        } else {
          metric->record(
              /* output= */ BoltVector::makeDenseVector({1.0, 0.0}),
              /* labels= */ labels->at(batch_idx)[vec_idx]);
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
  archive(_model, _emb, _freeze_hash_tables, _binary_prediction_threshold);
}

}  // namespace thirdai::automl::udt::utils