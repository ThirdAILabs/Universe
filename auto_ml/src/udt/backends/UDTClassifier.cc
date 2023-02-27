#include "UDTClassifier.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <bolt/src/graph/ExecutionConfig.h>
#include <auto_ml/src/cold_start/ColdStartDataSource.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/utils/Conversion.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <pybind11/stl.h>
#include <optional>
#include <stdexcept>
#include <variant>

namespace thirdai::automl::udt {

UDTClassifier::UDTClassifier(const data::ColumnDataTypes& input_data_types,
                             const data::UserProvidedTemporalRelationships&
                                 temporal_tracking_relationships,
                             const std::string& target_name,
                             data::CategoricalDataTypePtr target,
                             uint32_t n_target_classes, bool integer_target,
                             const data::TabularOptions& tabular_options,
                             const std::optional<std::string>& model_config,
                             const config::ArgumentMap& user_args) {
  if (model_config) {
    _model = utils::loadModel({tabular_options.feature_hash_range},
                              n_target_classes, *model_config);
  } else {
    uint32_t hidden_dim = user_args.get<uint32_t>(
        "embedding_dimension", "integer", defaults::HIDDEN_DIM);
    _model = utils::defaultModel(tabular_options.feature_hash_range, hidden_dim,
                                 n_target_classes);
  }

  bool normalize_target_categories = utils::hasSoftmaxOutput(_model);
  _label_block = labelBlock(target_name, target, n_target_classes,
                            integer_target, normalize_target_categories);

  bool force_parallel = user_args.get<bool>("force_parallel", "boolean", false);

  _dataset_factory = std::make_shared<data::TabularDatasetFactory>(
      input_data_types, temporal_tracking_relationships,
      std::vector<dataset::BlockPtr>{_label_block},
      std::set<std::string>{target_name}, tabular_options, force_parallel);

  _freeze_hash_tables = user_args.get<bool>("freeze_hash_tables", "boolean",
                                            defaults::FREEZE_HASH_TABLES);
}

void UDTClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<Validation>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  size_t batch_size = batch_size_opt.value_or(defaults::BATCH_SIZE);

  bolt::TrainConfig train_config = utils::getTrainConfig(
      epochs, learning_rate, validation, metrics, callbacks, verbose,
      logging_interval, _dataset_factory);

  auto train_dataset =
      _dataset_factory->getDatasetLoader(data, /* shuffle= */ true);

  utils::train(_model, train_dataset, train_config, batch_size,
               max_in_memory_batches,
               /* freeze_hash_tables= */ _freeze_hash_tables,
               licensing::TrainPermissionsToken(data->resourceName()));

  /**
   * For binary classification we tune the prediction threshold to optimize some
   * metric. This can improve performance particularly on datasets with a class
   * imbalance.
   */
  if (_model->outputDim() == 2) {
    if (validation && !validation->metrics().empty()) {
      validation->data()->restart();
      _binary_prediction_threshold =
          tuneBinaryClassificationPredictionThreshold(
              /* data_source= */ validation->data(),
              /* metric_name= */ validation->metrics().at(0), batch_size);

    } else if (!train_config.metrics().empty()) {
      data->restart();
      _binary_prediction_threshold =
          tuneBinaryClassificationPredictionThreshold(
              /* data_source= */ data,
              /* metric_name= */ train_config.metrics().at(0), batch_size);
    }
  }
}

py::object UDTClassifier::evaluate(const dataset::DataSourcePtr& data,
                                   const std::vector<std::string>& metrics,
                                   bool sparse_inference,
                                   bool return_predicted_class, bool verbose,
                                   bool return_metrics) {
  bolt::EvalConfig eval_config =
      utils::getEvalConfig(metrics, sparse_inference, verbose);

  auto [test_data, test_labels] =
      _dataset_factory->getDatasetLoader(data, /* shuffle= */ false)
          ->loadAll(/* batch_size= */ defaults::BATCH_SIZE, verbose);

  auto [output_metrics, output] =
      _model->evaluate(test_data, test_labels, eval_config);
  if (return_metrics) {
    return py::cast(output_metrics);
  }

  if (return_predicted_class) {
    return utils::predictedClasses(output, _binary_prediction_threshold);
  }

  return utils::convertInferenceTrackerToNumpy(output);
}

py::object UDTClassifier::predict(const MapInput& sample, bool sparse_inference,
                                  bool return_predicted_class) {
  BoltVector output = _model->predictSingle(
      _dataset_factory->featurizeInput(sample), sparse_inference);

  if (return_predicted_class) {
    return py::cast(
        utils::predictedClass(output, _binary_prediction_threshold));
  }

  return utils::convertBoltVectorToNumpy(output);
}

py::object UDTClassifier::predictBatch(const MapInputBatch& samples,
                                       bool sparse_inference,
                                       bool return_predicted_class) {
  BoltBatch outputs = _model->predictSingleBatch(
      _dataset_factory->featurizeInputBatch(samples), sparse_inference);

  if (return_predicted_class) {
    return utils::predictedClasses(outputs, _binary_prediction_threshold);
  }

  return utils::convertBoltBatchToNumpy(outputs);
}

std::vector<dataset::Explanation> UDTClassifier::explain(
    const MapInput& sample,
    const std::optional<std::variant<uint32_t, std::string>>& target_class) {
  std::optional<uint32_t> target_neuron = std::nullopt;
  if (target_class) {
    target_neuron = labelToNeuronId(*target_class);
  }

  auto [gradients_indices, gradients_ratio] = _model->getInputGradientSingle(
      /* input_data= */ {_dataset_factory->featurizeInput(sample)},
      /* explain_prediction_using_highest_activation= */ true,
      /* neuron_to_explain= */ target_neuron);
  auto explanation =
      _dataset_factory->explain(gradients_indices, gradients_ratio, sample);

  return explanation;
}

void UDTClassifier::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    const std::optional<Validation>& validation,
    const std::vector<bolt::CallbackPtr>& callbacks, bool verbose) {
  if (!integerTarget()) {
    throw std::invalid_argument(
        "Cold start pretraining currently only supports integer labels.");
  }

  if (_dataset_factory->inputDataTypes().size() != 1 ||
      !data::asText(_dataset_factory->inputDataTypes().begin()->second)) {
    throw std::invalid_argument(
        "Cold start pretraining can only be used on datasets with a single "
        "text input column and target column. The current model is configured "
        "with " +
        std::to_string(_dataset_factory->inputDataTypes().size()) +
        " input columns.");
  }

  std::string text_column_name =
      _dataset_factory->inputDataTypes().begin()->first;

  auto dataset = thirdai::data::ColumnMap::createStringColumnMapFromFile(
      data, _dataset_factory->delimiter());

  thirdai::data::ColdStartTextAugmentation augmentation(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* label_column_name= */ _label_block->columnName(),
      /* output_column_name= */ text_column_name);

  auto augmented_data = augmentation.apply(dataset);

  auto data_source = cold_start::ColdStartDataSource::make(
      /* column_map= */ augmented_data,
      /* text_column_name= */ text_column_name,
      /* label_column_name= */ _label_block->columnName(),
      /* column_delimiter= */ _dataset_factory->delimiter(),
      /* label_delimiter= */ _label_block->delimiter(),
      /* resource_name = */ data->resourceName());

  // TODO(david): reconsider validation. Instead of forcing users to pass in a
  // supervised dataset of query product pairs, can we create a synthetic
  // validation set based on the product catalog? This synthetic validation set
  // should NOT exactly model the cold start augmentation strategy but should
  // use a new strategy that can emulate real user queries without data leakage.
  // One idea here is to, for each product, generate a couple of fake user
  // queries which are just phrases of 3-4 consecutive words.

  train(data_source, learning_rate, epochs, validation,
        /* batch_size = */ std::nullopt,
        /* max_in_memory_batches= */ std::nullopt, metrics,
        /* callbacks= */ callbacks, /* verbose= */ verbose,
        /* logging_interval= */ std::nullopt);
}

py::object UDTClassifier::embedding(const MapInput& sample) {
  auto input_vector = _dataset_factory->featurizeInput(sample);
  BoltVector emb = _model->predictSingle(std::move(input_vector),
                                         /* use_sparse_inference= */ false,
                                         /* output_node_name= */ "fc_1");
  return utils::convertBoltVectorToNumpy(emb);
}

py::object UDTClassifier::entityEmbedding(
    const std::variant<uint32_t, std::string>& label) {
  uint32_t neuron_id = labelToNeuronId(label);

  auto fc_layers = _model->getNodes().back()->getInternalFullyConnectedLayers();

  if (fc_layers.size() != 1) {
    throw std::invalid_argument(
        "This UDT architecture currently doesn't support getting entity "
        "embeddings.");
  }

  auto weights = fc_layers.front()->getWeightsByNeuron(neuron_id);

  utils::NumpyArray<float> np_weights(weights.size());

  std::copy(weights.begin(), weights.end(), np_weights.mutable_data());

  return std::move(np_weights);
}

dataset::CategoricalBlockPtr UDTClassifier::labelBlock(
    const std::string& target_name, data::CategoricalDataTypePtr& target_config,
    uint32_t n_target_classes, bool integer_target,
    bool normalize_target_categories) {
  if (integer_target) {
    return dataset::NumericalCategoricalBlock::make(
        /* col= */ target_name,
        /* n_classes= */ n_target_classes,
        /* delimiter= */ target_config->delimiter,
        /* normalize_categories= */ normalize_target_categories);
  }

  _class_name_to_neuron = dataset::ThreadSafeVocabulary::make(
      /* vocab_size= */ n_target_classes);

  return dataset::StringLookupCategoricalBlock::make(
      /* col= */ target_name, /* vocab= */ _class_name_to_neuron,
      /* delimiter= */ target_config->delimiter,
      /* normalize_categories= */ normalize_target_categories);
}

uint32_t UDTClassifier::labelToNeuronId(
    const std::variant<uint32_t, std::string>& label) const {
  if (std::holds_alternative<uint32_t>(label)) {
    if (integerTarget()) {
      return std::get<uint32_t>(label);
    }
    throw std::invalid_argument(
        "Received an integer but integer_target is set to False (it is "
        "False by default). Target must be passed "
        "in as a string.");
  }
  if (std::holds_alternative<std::string>(label)) {
    if (!integerTarget()) {
      return _class_name_to_neuron->getUid(std::get<std::string>(label));
    }
    throw std::invalid_argument(
        "Received a string but integer_target is set to True. Target must be "
        "passed in as "
        "an integer.");
  }
  throw std::invalid_argument("Invalid entity type.");
}

std::optional<float> UDTClassifier::tuneBinaryClassificationPredictionThreshold(
    const dataset::DataSourcePtr& data_source, const std::string& metric_name,
    size_t batch_size) {
  // The number of samples used is capped to ensure tuning is fast even for
  // larger datasets.
  uint32_t num_batches =
      defaults::MAX_SAMPLES_FOR_THRESHOLD_TUNING / batch_size;

  auto dataset =
      _dataset_factory->getDatasetLoader(data_source, /* shuffle= */ true);

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

template void UDTClassifier::serialize(cereal::BinaryInputArchive&);
template void UDTClassifier::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void UDTClassifier::serialize(Archive& archive) {
  archive(cereal::base_class<UDTBackend>(this), _class_name_to_neuron,
          _label_block, _model, _dataset_factory, _freeze_hash_tables,
          _binary_prediction_threshold);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTClassifier)