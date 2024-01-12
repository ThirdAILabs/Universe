#include "Classifier.h"
#include <cereal/archives/binary.hpp>
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/python_bindings/NumpyConversions.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <pybind11/stl.h>
#include <optional>
#include <utility>

namespace thirdai::automl::udt::utils {

using bolt::metrics::fromMetricNames;

Classifier::Classifier(bolt::ModelPtr model, bool freeze_hash_tables)
    : _model(std::move(model)), _freeze_hash_tables(freeze_hash_tables) {
  if (_model->outputs().size() != 1) {
    throw std::invalid_argument(
        "Classifier utility is intended for single output models.");
  }

  auto computations = _model->computationOrder();

  // This defines the embedding as the second to last computatation in the
  // computation graph.
  // TODO(Nicholas): should this be configurable using the model config, and
  // have a default for the default model.
  _emb = computations.at(computations.size() - 2);
}

py::object Classifier::train(const dataset::DatasetLoaderPtr& dataset,
                             float learning_rate, uint32_t epochs,
                             const std::vector<std::string>& train_metrics,
                             const dataset::DatasetLoaderPtr& val_dataset,
                             const std::vector<std::string>& val_metrics,
                             const std::vector<CallbackPtr>& callbacks,
                             TrainOptions options,
                             const bolt::DistributedCommPtr& comm) {
  auto history = train(
      dataset, learning_rate, epochs,
      fromMetricNames(_model, train_metrics, /* prefix= */ "train_"),
      val_dataset, fromMetricNames(_model, val_metrics, /* prefix= */ "val_"),
      callbacks, options, comm);

  /**
   * For binary classification we tune the prediction threshold to optimize
   * some metric. This can improve performance particularly on datasets with
   * a class imbalance. We don't tune the threshold in the other method due to
   * how the metrics are passed in.
   */
  if (_model->outputs().at(0)->dim() == 2) {
    if (!val_metrics.empty()) {
      val_dataset->restart();
      _binary_prediction_threshold =
          tuneBinaryClassificationPredictionThreshold(
              /* dataset= */ val_dataset,
              /* metric_name= */ val_metrics.at(0));

    } else if (!train_metrics.empty()) {
      dataset->restart();
      _binary_prediction_threshold =
          tuneBinaryClassificationPredictionThreshold(
              /* dataset= */ dataset,
              /* metric_name= */ train_metrics.at(0));
    }
  }

  return history;
}

py::object Classifier::train(const dataset::DatasetLoaderPtr& dataset,
                             float learning_rate, uint32_t epochs,
                             const InputMetrics& train_metrics,
                             const dataset::DatasetLoaderPtr& val_dataset,
                             const InputMetrics& val_metrics,
                             const std::vector<CallbackPtr>& callbacks,
                             TrainOptions options,
                             const bolt::DistributedCommPtr& comm) {
  uint32_t batch_size = options.batch_size.value_or(defaults::BATCH_SIZE);

  std::optional<uint32_t> freeze_hash_tables_epoch = std::nullopt;
  if (_freeze_hash_tables) {
    freeze_hash_tables_epoch = 1;
  }

  bolt::Trainer trainer(_model, freeze_hash_tables_epoch,
                        /* gradient_update_interval */ 1,
                        bolt::python::CtrlCCheck{});

  auto history = trainer.train_with_dataset_loader(
      /* train_data_loader= */ dataset,
      /* learning_rate= */ learning_rate, /* epochs= */ epochs,
      /* batch_size= */ batch_size,
      /* max_in_memory_batches= */ options.max_in_memory_batches,
      /* train_metrics= */ train_metrics,
      /* validation_data_loader= */ val_dataset,
      /* validation_metrics= */ val_metrics,
      /* steps_per_validation= */ options.steps_per_validation,
      /* use_sparsity_in_validation= */ options.sparse_validation,
      /* callbacks= */ callbacks, /* autotune_rehash_rebuild= */ true,
      /* verbose= */ options.verbose,
      /* logging_interval= */ options.logging_interval,
      /*comm= */ comm);

  return py::cast(history);
}

py::object Classifier::train(const bolt::LabeledDataset& train_data,
                             float learning_rate, uint32_t epochs,
                             const InputMetrics& train_metrics,
                             const bolt::LabeledDataset& val_data,
                             const InputMetrics& val_metrics,
                             const std::vector<CallbackPtr>& callbacks,
                             TrainOptions options) {
  // We first need to verify whether the tensors are of appropriate dimensions
  // or not
  auto vec_eq = [](const auto& a, const auto& b) -> bool {
    if (a.size() != b.size()) {
      return false;
    }
    for (uint32_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  };

  auto get_tensor_size =
      [](const bolt::Dataset& dataset) -> std::vector<uint32_t> {
    std::vector<uint32_t> dims;
    for (const auto& tensor : dataset[0]) {
      dims.push_back(tensor->dim());
    }
    return dims;
  };

  auto assert_valid_tensors = [get_tensor_size, vec_eq](
                                  const bolt::ModelPtr& model,
                                  const bolt::LabeledDataset& dataset) -> void {
    auto model_input_dims = model->inputDims();
    auto input_tensor_dims = get_tensor_size(dataset.first);

    if (!vec_eq(model_input_dims, input_tensor_dims)) {
      throw std::invalid_argument(
          "Input dim mismatch while training on tensors.");
    }

    auto model_output_dims = model->labelDims();
    auto output_tensor_dims = get_tensor_size(dataset.second);
    if (!vec_eq(model_output_dims, output_tensor_dims)) {
      throw std::invalid_argument(
          "Label dim mismatch while training on tensors.");
    }
  };
  auto print_vector = [](const std::vector<uint32_t>& vec) -> void {
    for (const auto& x : vec) {
      std::cout << x << ", ";
    }
    std::cout << std::endl;
  };

  assert_valid_tensors(_model, train_data);
  assert_valid_tensors(_model, val_data);

  std::cout << "Input Dims: ";
  print_vector(_model->inputDims());

  std::cout << "Data Input Dims: ";
  print_vector(get_tensor_size(train_data.first));

  std::cout << "Output Dims: ";
  print_vector(_model->labelDims());

  std::cout << "Data Label Dims: ";
  print_vector(get_tensor_size(train_data.second));

  std::optional<uint32_t> freeze_hash_tables_epoch = std::nullopt;
  if (_freeze_hash_tables) {
    freeze_hash_tables_epoch = 1;
  }
  bolt::Trainer trainer(_model, freeze_hash_tables_epoch,
                        /* gradient_update_interval */ 1,
                        bolt::python::CtrlCCheck{});
  auto history = trainer.train(
      /* train_data= */ train_data,
      /* learning_rate= */ learning_rate, /* epochs= */ epochs,
      /* train_metrics= */ train_metrics,
      /* validation_data= */ val_data,
      /* validation_metrics= */ val_metrics,
      /* steps_per_validation= */ options.steps_per_validation,
      /* use_sparsity_in_validation= */ options.sparse_validation,
      /* callbacks= */ callbacks, /* autotune_rehash_rebuild= */ true,
      /* verbose= */ options.verbose,
      /* logging_interval= */ options.logging_interval);

  return py::cast(history);
}

py::object Classifier::evaluate(dataset::DatasetLoaderPtr& dataset,
                                const std::vector<std::string>& metrics,
                                bool sparse_inference, bool verbose) {
  return evaluate(dataset,
                  fromMetricNames(_model, metrics, /* prefix= */ "val_"),
                  sparse_inference, verbose);
}

py::object Classifier::evaluate(dataset::DatasetLoaderPtr& dataset,
                                const InputMetrics& metrics,
                                bool sparse_inference, bool verbose) {
  bolt::Trainer trainer(_model, std::nullopt, /* gradient_update_interval */ 1,
                        bolt::python::CtrlCCheck{});

  auto history = trainer.validate_with_dataset_loader(
      dataset, metrics, sparse_inference, verbose);

  return py::cast(history);
}

py::object Classifier::predict(const bolt::TensorList& inputs,
                               bool sparse_inference,
                               bool return_predicted_class, bool single,
                               std::optional<uint32_t> top_k) {
  auto output = _model->forward(inputs, sparse_inference).at(0);
  if (return_predicted_class) {
    return predictedClasses(output, single);
  }

  auto nonzeros = output->nonzeros();

  if (!nonzeros) {
    throw std::runtime_error("Number of nonzeros is not fixed");
  }
  if (top_k) {
    if (top_k.value() > *nonzeros || (top_k.value() == 0)) {
      if (output->activeNeuronsPtr()) {
        throw std::runtime_error(
            "top_k value is invalid.  top_k <= number of target "
            "classes * sparsity");
      }
      throw std::runtime_error(
          "top_k value is invalid. top_k > 0 and top_k <= number of target "
          "classes");
    }
    return bolt::python::tensorToNumpyTopK(output,
                                           /* single_row_to_vector= */ single,
                                           top_k.value());
  }
  return bolt::python::tensorToNumpy(output,
                                     /* single_row_to_vector= */ single);
}

py::object Classifier::embedding(const bolt::TensorList& inputs) {
  // TODO(Nicholas): Sparsity could speed this up, and wouldn't affect the
  // embeddings if the sparsity is in the output layer and the embeddings are
  // from the hidden layer.
  _model->forward(inputs, /* use_sparsity= */ false);

  return bolt::python::tensorToNumpy(_emb->tensor());
}

uint32_t Classifier::predictedClass(const BoltVector& output) {
  if (_binary_prediction_threshold) {
    return output.activations[1] >= *_binary_prediction_threshold;
  }
  return output.getHighestActivationId();
}

py::object Classifier::predictedClasses(const bolt::TensorPtr& output,
                                        bool single) {
  if (output->batchSize() == 1 && single) {
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
  auto tensor_batches = bolt::convertDatasets(dataset, _model->inputDims());

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

  double best_metric_value = bolt_v1::makeMetric(metric_name)->worst();
  std::optional<float> best_threshold = std::nullopt;

#pragma omp parallel for default(none) \
    shared(labels, best_metric_value, best_threshold, metric_name, scores)
  for (uint32_t t_idx = 1; t_idx < defaults::NUM_THRESHOLDS_TO_CHECK; t_idx++) {
    // TODO(Nicholas): This is still using the old metric from bolt v1. The bolt
    // v2 metrics are more intertwined with the computation graph and would be
    // harder to use in this way.
    auto metric = bolt_v1::makeMetric(metric_name);

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
