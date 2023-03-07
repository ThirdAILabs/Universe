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
#include <licensing/src/CheckLicense.h>
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
                             const config::ArgumentMap& user_args)
    : _classifier(utils::buildModel(
                      /* input_dim= */ tabular_options.feature_hash_range,
                      /* output_dim= */ n_target_classes,
                      /* args= */ user_args, /* model_config= */ model_config),
                  user_args.get<bool>("freeze_hash_tables", "boolean",
                                      defaults::FREEZE_HASH_TABLES)) {
  bool normalize_target_categories = utils::hasSoftmaxOutput(model());
  _label_block = labelBlock(target_name, target, n_target_classes,
                            integer_target, normalize_target_categories);

  bool force_parallel = user_args.get<bool>("force_parallel", "boolean", false);

  _dataset_factory = std::make_shared<data::TabularDatasetFactory>(
      input_data_types, temporal_tracking_relationships,
      std::vector<dataset::BlockPtr>{_label_block},
      std::set<std::string>{target_name}, tabular_options, force_parallel);
}

void UDTClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<ValidationDataSource>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  std::optional<ValidationDatasetLoader> validation_dataset_loader =
      std::nullopt;
  if (validation) {
    validation_dataset_loader =
        ValidationDatasetLoader(_dataset_factory->getDatasetLoader(
                                    validation->first, /* shuffle= */ false),
                                validation->second);
  }

  auto train_dataset_loader =
      _dataset_factory->getDatasetLoader(data, /* shuffle= */ true);

  _classifier.train(train_dataset_loader, learning_rate, epochs,
                    validation_dataset_loader, batch_size_opt,
                    max_in_memory_batches, metrics, callbacks, verbose,
                    logging_interval, licensing::TrainPermissionsToken(data));
}

py::object UDTClassifier::evaluate(const dataset::DataSourcePtr& data,
                                   const std::vector<std::string>& metrics,
                                   bool sparse_inference,
                                   bool return_predicted_class, bool verbose,
                                   bool return_metrics,
                                   std::optional<uint32_t> top_k) {
  (void)top_k;
  auto dataset = _dataset_factory->getDatasetLoader(data, /* shuffle= */ false);

  return _classifier.evaluate(dataset, metrics, sparse_inference,
                              return_predicted_class, verbose, return_metrics);
}

py::object UDTClassifier::predict(const MapInput& sample, bool sparse_inference,
                                  bool return_predicted_class,
                                  std::optional<uint32_t> top_k) {
  (void)top_k;
  return _classifier.predict(_dataset_factory->featurizeInput(sample),
                             sparse_inference, return_predicted_class);
}

py::object UDTClassifier::predictBatch(const MapInputBatch& samples,
                                       bool sparse_inference,
                                       bool return_predicted_class,
                                       std::optional<uint32_t> top_k) {
  (void)top_k;
  return _classifier.predictBatch(
      _dataset_factory->featurizeInputBatch(samples), sparse_inference,
      return_predicted_class);
}

std::vector<dataset::Explanation> UDTClassifier::explain(
    const MapInput& sample,
    const std::optional<std::variant<uint32_t, std::string>>& target_class) {
  std::optional<uint32_t> target_neuron = std::nullopt;
  if (target_class) {
    target_neuron = labelToNeuronId(*target_class);
  }

  auto [gradients_indices, gradients_ratio] =
      _classifier.model()->getInputGradientSingle(
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
    const std::optional<ValidationDataSource>& validation,
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
  BoltVector emb =
      _classifier.model()->predictSingle(std::move(input_vector),
                                         /* use_sparse_inference= */ false,
                                         /* output_node_name= */ "fc_1");
  return utils::convertBoltVectorToNumpy(emb);
}

py::object UDTClassifier::entityEmbedding(
    const std::variant<uint32_t, std::string>& label) {
  uint32_t neuron_id = labelToNeuronId(label);

  auto fc_layers =
      _classifier.model()->getNodes().back()->getInternalFullyConnectedLayers();

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
      /* max_vocab_size= */ n_target_classes);

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

template void UDTClassifier::serialize(cereal::BinaryInputArchive&);
template void UDTClassifier::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void UDTClassifier::serialize(Archive& archive) {
  archive(cereal::base_class<UDTBackend>(this), _class_name_to_neuron,
          _label_block, _classifier, _dataset_factory);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTClassifier)