#include "UDTClassifier.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <auto_ml/src/cold_start/ColdStartDataSource.h>
#include <auto_ml/src/cold_start/ColdStartUtils.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/udt/utils/Conversion.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <stdexcept>
#include <variant>

namespace thirdai::automl::udt {

UDTClassifier::UDTClassifier(
    const data::ColumnDataTypes& input_data_types,
    data::UserProvidedTemporalRelationships temporal_tracking_relationships,
    const std::string& target_name, data::CategoricalDataTypePtr target,
    uint32_t n_target_classes, bool integer_target,
    std::string time_granularity, uint32_t lookahead, char delimiter,
    const config::ArgumentMap& options) {
  data::tabular::TabularBlockOptions tabular_options;

  tabular_options.contextual_columns =
      options.get<bool>("contextual_columns", "boolean", false);
  tabular_options.time_granularity = std::move(time_granularity);
  tabular_options.lookahead = lookahead;

  bool normalize_target_categories =
      options.get<bool>("normalize_target_categories", "boolean", false);

  _label_block = labelBlock(target_name, target, n_target_classes,
                            integer_target, normalize_target_categories);

  bool force_parallel = options.get<bool>("force_parallel", "boolean", false);

  _dataset_factory = std::make_shared<data::tabular::TabularDatasetFactory>(
      input_data_types, temporal_tracking_relationships,
      std::vector<dataset::BlockPtr>{_label_block}, tabular_options, delimiter,
      force_parallel);

  uint32_t hidden_dim = options.get<uint32_t>("embedding_dim", "integer", 512);

  _model = utils::defaultModel(_dataset_factory->inputDim(), hidden_dim,
                               n_target_classes);

  _freeze_hash_tables =
      options.get<bool>("freeze_hash_tables", "boolean", true);
}

void UDTClassifier::train(
    const dataset::DataSourcePtr& train_data, uint32_t epochs,
    float learning_rate, const std::optional<Validation>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& train_metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  size_t batch_size = batch_size_opt.value_or(2048);

  bolt::TrainConfig train_config = utils::getTrainConfig(
      epochs, learning_rate, validation, train_metrics, callbacks, verbose,
      logging_interval, _dataset_factory);

  auto train_dataset =
      _dataset_factory->getDatasetLoader(train_data, /* training= */ true);

  utils::train(_model, train_dataset, train_config, batch_size,
               max_in_memory_batches,
               /* freeze_hash_tables= */ _freeze_hash_tables);
}

py::object UDTClassifier::evaluate(const dataset::DataSourcePtr& data,
                                   const std::vector<std::string>& metrics,
                                   bool sparse_inference,
                                   bool return_predicted_class, bool verbose) {
  bolt::EvalConfig eval_config =
      utils::getEvalConfig(metrics, sparse_inference, verbose);

  auto [test_data, test_labels] =
      _dataset_factory->getDatasetLoader(data, /* training= */ false)
          ->loadAll(/* batch_size= */ 2048, verbose);

  auto [_, output] = _model->evaluate(test_data, test_labels, eval_config);

  if (return_predicted_class) {
    utils::NumpyArray<uint32_t> predictions(output.numSamples());
    for (uint32_t i = 0; i < output.numSamples(); i++) {
      predictions.mutable_at(i) =
          output.getSampleAsNonOwningBoltVector(i).getHighestActivationId();
    }
    return py::object(std::move(predictions));
  }

  return utils::convertInferenceTrackerToNumpy(output);
}

py::object UDTClassifier::predict(const MapInput& sample, bool sparse_inference,
                                  bool return_predicted_class) {
  BoltVector output = _model->predictSingle(
      _dataset_factory->featurizeInput(sample), sparse_inference);

  if (return_predicted_class) {
    return py::cast(output.getHighestActivationId());
  }

  return utils::convertBoltVectorToNumpy(output);
}

py::object UDTClassifier::predictBatch(const MapInputBatch& samples,
                                       bool sparse_inference,
                                       bool return_predicted_class) {
  BoltBatch outputs = _model->predictSingleBatch(
      _dataset_factory->featurizeInputBatch(samples), sparse_inference);

  if (return_predicted_class) {
    utils::NumpyArray<uint32_t> predictions(outputs.getBatchSize());
    for (uint32_t i = 0; i < outputs.getBatchSize(); i++) {
      predictions.mutable_at(i) = outputs[i].getHighestActivationId();
    }
    return py::object(std::move(predictions));
  }

  return utils::convertBoltBatchToNumpy(outputs);
}

std::vector<dataset::Explanation> UDTClassifier::explain(
    const MapInput& sample,
    const std::optional<std::variant<uint32_t, std::string>>& target_class) {
  std::optional<uint32_t> target_neuron;
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
    const dataset::DataSourcePtr& original_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, uint32_t epochs,
    float learning_rate, const std::vector<std::string>& train_metrics,
    const std::optional<Validation>& validation, bool verbose) {
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
      original_source, _dataset_factory->delimiter());

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
      /* resource_name = */ original_source->resourceName());

  // TODO(david): reconsider validation. Instead of forcing users to pass in a
  // supervised dataset of query product pairs, can we create a synthetic
  // validation set based on the product catalog? This synthetic validation set
  // should NOT exactly model the cold start augmentation strategy but should
  // use a new strategy that can emulate real user queries without data leakage.
  // One idea here is to, for each product, generate a couple of fake user
  // queries which are just phrases of 3-4 consecutive words.

  train(data_source, epochs, learning_rate, validation,
        /* batch_size = */ std::nullopt,
        /* max_in_memory_batches= */ std::nullopt, train_metrics,
        /* callbacks= */ {}, verbose,
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
  if (std::holds_alternative<uint32_t>(label) && integerTarget()) {
    return std::get<uint32_t>(label);
  }
  if (!integerTarget()) {
    return _class_name_to_neuron->getUid(std::get<std::string>(label));
  }
  throw std::invalid_argument("Invalid entity type.");
}

}  // namespace thirdai::automl::udt