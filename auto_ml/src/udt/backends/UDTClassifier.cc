#include "UDTClassifier.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <auto_ml/src/udt/utils/Conversion.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
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
  std::vector<dataset::BlockPtr> label_block = {
      labelBlock(target_name, target, n_target_classes, integer_target,
                 normalize_target_categories)};

  bool parallel = options.get<bool>("parallel", "boolean", false);

  _dataset_factory = std::make_shared<data::tabular::TabularDatasetFactory>(
      input_data_types, temporal_tracking_relationships, label_block,
      tabular_options, delimiter, parallel);

  uint32_t hidden_dim = options.get<uint32_t>("embedding_dim", "integer", 512);

  _model = utils::defaultModel(_dataset_factory->inputDim(), hidden_dim,
                               n_target_classes);
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
               /* freeze_hash_tables= */ true);
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

py::object UDTClassifier::embedding(const MapInput& sample) {
  auto input_vector = _dataset_factory->featurizeInput(sample);
  BoltVector emb = _model->predictSingle(std::move(input_vector),
                                         /* use_sparse_inference= */ false,
                                         /* output_node_name= */ "fc_1");
  return utils::convertBoltVectorToNumpy(emb);
}

py::object UDTClassifier::entityEmbedding(
    const std::variant<uint32_t, std::string>& label) {
  uint32_t label_id;
  if (std::holds_alternative<uint32_t>(label) && !_class_name_to_neuron) {
    label_id = std::get<uint32_t>(label);
  } else if (_class_name_to_neuron) {
    label_id = _class_name_to_neuron->getUid(std::get<std::string>(label));
  } else {
    throw std::invalid_argument("Invalid entity type.");
  }

  auto fc_layers = _model->getNodes().back()->getInternalFullyConnectedLayers();

  if (fc_layers.size() != 1) {
    throw std::invalid_argument(
        "This UDT architecture currently doesn't support getting entity "
        "embeddings.");
  }

  auto weights = fc_layers.front()->getWeightsByNeuron(label_id);

  utils::NumpyArray<float> np_weights(weights.size());

  std::copy(weights.begin(), weights.end(), np_weights.mutable_data());

  return std::move(np_weights);
}

dataset::BlockPtr UDTClassifier::labelBlock(
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

}  // namespace thirdai::automl::udt