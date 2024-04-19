#include "UDTPretrainedText.h"
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <archive/src/Archive.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/SpladeMachAugmentation.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringIDLookup.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/cold_start/ColdStartText.h>
#include <data/src/transformations/cold_start/VariableLengthColdStart.h>
#include <limits>
#include <memory>
#include <stdexcept>

namespace thirdai::automl::udt {

static const std::string LABEL_VOCAB = "__label_vocab__";

std::pair<std::string, TextDataTypePtr> textDataType(
    const ColumnDataTypes& data_types) {
  if (data_types.size() != 2) {
    throw std::invalid_argument(
        "Expected only a text input and categorial output to use pretrained "
        "classifier.");
  }

  for (const auto& [name, type] : data_types) {
    if (auto text = asText(type)) {
      return {name, text};
    }
  }

  throw std::invalid_argument(
      "Expected only a text input and categorial output to use pretrained "
      "classifier.");
}

std::pair<std::string, CategoricalDataTypePtr> categoricalDataType(
    const ColumnDataTypes& data_types) {
  if (data_types.size() != 2) {
    throw std::invalid_argument(
        "Expected only a text input and categorial output to use pretrained "
        "classifier.");
  }

  for (const auto& [name, type] : data_types) {
    if (auto cat = asCategorical(type)) {
      return {name, cat};
    }
  }

  throw std::invalid_argument(
      "Expected only a text input and categorial output to use pretrained "
      "classifier.");
}

uint32_t getInputDim(const config::ArgumentMap& user_args) {
  uint32_t input_dim = user_args.get<uint32_t>("input_dim", "integer",
                                               defaults::FEATURE_HASH_RANGE);
  if (user_args.contains("fhr")) {
    // For the QT app distribution we want to be able to override the input_dim
    // without revealing any information about the architecture.
    input_dim = user_args.get<uint32_t>("fhr", "integer");
  }

  return input_dim;
}

UDTPretrainedText::UDTPretrainedText(const ColumnDataTypes& data_types,
                                     uint32_t n_target_classes,
                                     bool integer_target,
                                     const SpladeMachPtr& pretrained_model,
                                     char delimiter,
                                     const config::ArgumentMap& user_args)
    : _classifier(utils::Classifier::make(
          utils::buildModel(
              /* input_dim= */ getInputDim(user_args),
              /* output_dim= */ n_target_classes,
              /* args= */ user_args, /* model_config= */ std::nullopt,
              /* use_sigmoid_bce = */
              user_args.get<bool>("sigmoid_bce", "boolean",
                                  defaults::USE_SIGMOID_BCE)),
          user_args.get<bool>("freeze_hash_tables", "boolean",
                              defaults::FREEZE_HASH_TABLES))),
      _state(std::make_shared<data::State>()),
      _delimiter(delimiter) {
  auto [text_col, text_type] = textDataType(data_types);
  auto [cat_col, cat_type] = categoricalDataType(data_types);
  _text_column = text_col;

  _pretrained_augmentation = nullptr;
  if (pretrained_model) {
    _pretrained_augmentation = std::make_shared<data::SpladeMachAugmentation>(
        text_col, SPLADE_TOKENS, pretrained_model,
        user_args.get<uint32_t>("hashes_per_model", "integer", 10));
  }

  _text_transform = textTransformation(text_col, text_type,
                                       _classifier->model()->inputDims().at(0),
                                       !!_pretrained_augmentation);

  _label_transform =
      labelTransformation(cat_col, cat_type, n_target_classes, integer_target);

  _bolt_inputs = {data::OutputColumns(FEATURIZED_INDICES, FEATURIZED_VALUES)};

  bool softmax = utils::hasSoftmaxOutput(_classifier->model());

  _bolt_labels = {data::OutputColumns(
      FEATURIZED_LABELS,
      softmax ? data::ValueFillType::SumToOne : data::ValueFillType::Ones)};
}

py::object UDTPretrainedText::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm, py::kwargs kwargs) {
  (void)kwargs;

  auto train_data_loader = getDataLoader(
      buildPipeline(), data, options.batchSize(), /* shuffle= */ true,
      options.verbose, options.shuffle_config);

  data::LoaderPtr val_data_loader;
  if (val_data) {
    val_data_loader =
        getDataLoader(buildPipeline(), val_data, defaults::BATCH_SIZE,
                      /* shuffle= */ false, options.verbose);
  }

  return _classifier->train(train_data_loader, learning_rate, epochs,
                            train_metrics, val_data_loader, val_metrics,
                            callbacks, options, comm);
}

py::object UDTPretrainedText::evaluate(const dataset::DataSourcePtr& data,
                                       const std::vector<std::string>& metrics,
                                       bool sparse_inference, bool verbose,
                                       py::kwargs kwargs) {
  (void)kwargs;

  auto dataset = getDataLoader(buildPipeline(), data, defaults::BATCH_SIZE,
                               /* shuffle= */ false, verbose);

  return _classifier->evaluate(dataset, metrics, sparse_inference, verbose);
}

py::object UDTPretrainedText::predict(const MapInput& sample,
                                      bool sparse_inference,
                                      bool return_predicted_class,
                                      std::optional<uint32_t> top_k) {
  auto columns = data::ColumnMap::fromMapInput(sample);

  return predict(std::move(columns), sparse_inference, return_predicted_class,
                 top_k, /*single=*/true);
}

py::object UDTPretrainedText::predictBatch(const MapInputBatch& sample,
                                           bool sparse_inference,
                                           bool return_predicted_class,
                                           std::optional<uint32_t> top_k) {
  auto columns = data::ColumnMap::fromMapInputBatch(sample);

  return predict(std::move(columns), sparse_inference, return_predicted_class,
                 top_k, /*single=*/false);
}

py::object UDTPretrainedText::predict(data::ColumnMap columns,
                                      bool sparse_inference,
                                      bool return_predicted_class,
                                      std::optional<uint32_t> top_k,
                                      bool single) {
  if (_pretrained_augmentation) {
    columns = _pretrained_augmentation->applyStateless(std::move(columns));
  }
  columns = _text_transform->applyStateless(std::move(columns));

  auto tensors = data::toTensors(columns, _bolt_inputs);

  return _classifier->predict(tensors, sparse_inference, return_predicted_class,
                              /*single*/ single, top_k);
}

py::object UDTPretrainedText::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<data::VariableLengthConfig> variable_length,
    float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm, const py::kwargs& kwargs) {
  (void)kwargs;

  auto train_data_loader = getDataLoader(
      buildPipeline(strong_column_names, weak_column_names, variable_length),
      data, options.batchSize(), /* shuffle= */ true, options.verbose,
      options.shuffle_config);

  data::LoaderPtr val_data_loader;
  if (val_data) {
    val_data_loader =
        getDataLoader(buildPipeline(), val_data, defaults::BATCH_SIZE,
                      /* shuffle= */ false, options.verbose);
  }

  return _classifier->train(train_data_loader, learning_rate, epochs,
                            train_metrics, val_data_loader, val_metrics,
                            callbacks, options, comm);
}

std::string UDTPretrainedText::className(uint32_t class_id) const {
  if (!_state->containsVocab(LABEL_VOCAB)) {
    return std::to_string(class_id);
  }
  auto& vocab = _state->getVocab(LABEL_VOCAB);
  return vocab->getString(class_id);
}

data::TransformationPtr UDTPretrainedText::buildPipeline(
    const std::vector<std::string>& strong_cols,
    const std::vector<std::string>& weak_cols,
    std::optional<data::VariableLengthConfig> vlc) {
  auto pipeline = data::Pipeline::make();

  if (_pretrained_augmentation) {
    pipeline = pipeline->then(_pretrained_augmentation);
  }

  if (!strong_cols.empty() || !weak_cols.empty()) {
    if (vlc) {
      pipeline = pipeline->then(std::make_shared<data::VariableLengthColdStart>(
          /* strong_column_names= */ strong_cols,
          /* weak_column_names= */ weak_cols,
          /* output_column_name= */ _text_column,
          /* config= */ *vlc));
    } else {
      pipeline =
          pipeline->then(std::make_shared<data::ColdStartTextAugmentation>(
              /* strong_column_names= */ strong_cols,
              /* weak_column_names= */ weak_cols,
              /* output_column_name= */ _text_column));
    }
  }

  return pipeline->then(_text_transform)->then(_label_transform);
}

data::LoaderPtr UDTPretrainedText::getDataLoader(
    const data::TransformationPtr& transform,
    const dataset::DataSourcePtr& data_source, size_t batch_size, bool shuffle,
    bool verbose, dataset::DatasetShuffleConfig shuffle_config) {
  auto data_iter = data::CsvIterator::make(data_source, _delimiter);

  return data::Loader::make(
      data_iter, transform, _state, _bolt_inputs, _bolt_labels,
      /* batch_size= */ batch_size, /* shuffle= */ shuffle,
      /* verbose= */ verbose,
      /* shuffle_buffer_size= */ shuffle_config.min_buffer_size,
      /* shuffle_seed= */ shuffle_config.seed);
}

data::TransformationPtr UDTPretrainedText::textTransformation(
    const std::string& text_col, const TextDataTypePtr& text_type,
    uint32_t token_dim, bool has_augmentation) {
  if (!has_augmentation) {
    return std::make_shared<data::TextTokenizer>(
        text_col, FEATURIZED_INDICES, FEATURIZED_VALUES, text_type->tokenizer,
        text_type->encoder, text_type->lowercase, token_dim);
  }

  auto tokenizer = std::make_shared<data::TextTokenizer>(
      text_col, text_col, std::nullopt, text_type->tokenizer,
      text_type->encoder, text_type->lowercase,
      std::numeric_limits<uint32_t>::max());

  auto feature_hash = std::make_shared<data::FeatureHash>(
      std::vector<std::string>{text_col, SPLADE_TOKENS}, FEATURIZED_INDICES,
      FEATURIZED_VALUES, token_dim);

  return data::Pipeline::make({tokenizer, feature_hash});
}

data::TransformationPtr UDTPretrainedText::labelTransformation(
    const std::string& target_name, CategoricalDataTypePtr& target_config,
    uint32_t n_target_classes, bool integer_target) {
  if (integer_target) {
    if (!target_config->delimiter) {
      return std::make_shared<data::StringToToken>(
          target_name, FEATURIZED_LABELS, n_target_classes);
    }
    return std::make_shared<data::StringToTokenArray>(
        target_name, FEATURIZED_LABELS, target_config->delimiter.value(),
        n_target_classes);
  }

  return std::make_shared<data::StringIDLookup>(target_name, FEATURIZED_LABELS,
                                                LABEL_VOCAB, n_target_classes,
                                                target_config->delimiter);
}

ar::ConstArchivePtr UDTPretrainedText::toArchive(bool with_optimizer) const {
  auto map = _classifier->toArchive(with_optimizer);
  map->set("type", ar::str(type()));

  if (_pretrained_augmentation) {
    map->set("pretrained_augmentation", _pretrained_augmentation->toArchive());
  }
  map->set("text_transform", _text_transform->toArchive());
  map->set("label_transform", _label_transform->toArchive());

  map->set("state", _state->toArchive());

  map->set("bolt_inputs", data::outputColumnsToArchive(_bolt_inputs));
  map->set("bolt_labels", data::outputColumnsToArchive(_bolt_labels));

  map->set("text_column", ar::str(_text_column));
  map->set("delimiter", ar::character(_delimiter));

  return map;
}

UDTPretrainedText::UDTPretrainedText(const ar::Archive& archive)
    : _classifier(utils::Classifier::fromArchive(archive)),
      _text_transform(
          data::Transformation::fromArchive(*archive.get("text_transform"))),
      _label_transform(
          data::Transformation::fromArchive(*archive.get("label_transform"))),
      _state(data::State::fromArchive(*archive.get("state"))),
      _bolt_inputs(data::outputColumnsFromArchive(*archive.get("bolt_inputs"))),
      _bolt_labels(data::outputColumnsFromArchive(*archive.get("bolt_labels"))),
      _text_column(archive.str("text_column")),
      _delimiter(archive.getAs<ar::Char>("delimiter")) {
  if (archive.contains("pretrained_augmentation")) {
    _pretrained_augmentation = data::Transformation::fromArchive(
        *archive.get("pretrained_augmentation"));
  }
}

std::unique_ptr<UDTPretrainedText> UDTPretrainedText::fromArchive(
    const ar::Archive& archive) {
  return std::make_unique<UDTPretrainedText>(archive);
}

}  // namespace thirdai::automl::udt