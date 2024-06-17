#include "NerUDTModel.h"
#include <bolt/src/NER/Defaults.h>
#include <bolt/src/NER/model/NER.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <archive/src/Archive.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/ner/NerDyadicDataProcessor.h>
#include <data/src/transformations/ner/NerTokenizationUnigram.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt::NER {

void NerUDTModel::initializeNER(uint32_t fhr, uint32_t number_labels) {
  auto train_transforms = getTransformations(false, fhr, number_labels);

  auto inference_transforms = getTransformations(true, fhr, number_labels);

  auto bolt_inputs = {data::OutputColumns(_featurized_tokens_indices_column)};

  _classifier = std::make_shared<NerClassifier>(
      _bolt_model, bolt_inputs, train_transforms, inference_transforms,
      _tokens_column, _tags_column);
}

bolt::ModelPtr NerUDTModel::initializeBoltModel(
    uint32_t input_dim, uint32_t emb_dim, uint32_t output_dim,
    std::optional<std::vector<std::vector<float>*>> pretrained_emb) {
  auto input = bolt::Input::make(input_dim);

  auto emb_op = bolt::Embedding::make(emb_dim, input_dim, "relu",
                                      /* bias= */ true);
  if (pretrained_emb) {
    emb_op->setEmbeddings(pretrained_emb.value()[0]->data());
    emb_op->setBiases(pretrained_emb.value()[1]->data());
  }
  auto hidden = emb_op->apply(input);

  auto output =
      bolt::FullyConnected::make(output_dim, hidden->dim(), 1, "softmax",
                                 /* sampling= */ nullptr, /* use_bias= */ true)
          ->apply(hidden);

  auto labels = bolt::Input::make(output_dim);
  auto loss = bolt::CategoricalCrossEntropy::make(output, labels);

  return bolt::Model::make({input}, {output}, {loss});
}

NerUDTModel::NerUDTModel(
    bolt::ModelPtr model, std::string tokens_column, std::string tags_column,
    std::unordered_map<std::string, uint32_t> tag_to_label,
    std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
    std::optional<data::FeatureEnhancementConfig> feature_enhancement_config)
    : _bolt_model(std::move(model)),
      _tokens_column(std::move(tokens_column)),
      _tags_column(std::move(tags_column)),
      _target_word_tokenizers(std::move(target_word_tokenizers)),
      _tag_to_label(std::move(tag_to_label)),
      _feature_enhancement_config(std::move(feature_enhancement_config)),
      _featurized_sentence_column("featurized_sentence_for_" + tokens_column) {
  auto input_dims = _bolt_model->inputDims();
  if (input_dims.size() != 1) {
    throw std::logic_error(
        "Can only train a bolt model with a Single Input. Found model with "
        "number of inputs: " +
        std::to_string(input_dims.size()));
  }

  uint32_t fhr = input_dims[0];
  uint32_t number_labels = getMaxLabelFromTagToLabel(_tag_to_label);

  for (const auto& [k, v] : _tag_to_label) {
    _label_to_tag_map[v] = k;
  }
  initializeNER(fhr, number_labels);
}

NerUDTModel::NerUDTModel(
    std::string tokens_column, std::string tags_column,
    std::unordered_map<std::string, uint32_t> tag_to_label,
    std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
    std::optional<data::FeatureEnhancementConfig> feature_enhancement_config)
    : _tokens_column(std::move(tokens_column)),
      _tags_column(std::move(tags_column)),
      _target_word_tokenizers(std::move(target_word_tokenizers)),
      _tag_to_label(std::move(tag_to_label)),
      _feature_enhancement_config(std::move(feature_enhancement_config)),
      _featurized_sentence_column("featurized_sentence_for_" + tokens_column) {
  uint32_t number_labels = getMaxLabelFromTagToLabel(_tag_to_label);
  _bolt_model = initializeBoltModel(defaults::UDT_FEATURE_HASH_RANGE,
                                    defaults::UDT_EMB_DIM, number_labels);

  for (const auto& [k, v] : _tag_to_label) {
    _label_to_tag_map[v] = k;
  }
  initializeNER(defaults::UDT_FEATURE_HASH_RANGE, number_labels);
}

NerUDTModel::NerUDTModel(std::shared_ptr<NerUDTModel>& pretrained_model,
                         std::string tokens_column, std::string tags_column,
                         std::unordered_map<std::string, uint32_t> tag_to_label,
                         const std::optional<data::FeatureEnhancementConfig>&
                             feature_enhancement_config)
    : _tokens_column(std::move(tokens_column)),
      _tags_column(std::move(tags_column)),
      _target_word_tokenizers(pretrained_model->getTargetWordTokenizers()),
      _tag_to_label(std::move(tag_to_label)),
      _featurized_sentence_column("featurized_sentence_for_" + tokens_column) {
  if (feature_enhancement_config.has_value()) {
    _feature_enhancement_config = feature_enhancement_config;
  } else {
    _feature_enhancement_config = pretrained_model->_feature_enhancement_config;
  }
  uint32_t fhr = (pretrained_model->getBoltModel()->inputDims()[0]);
  uint32_t number_labels = getMaxLabelFromTagToLabel(_tag_to_label);

  auto emb_op = pretrained_model->getBoltModel()->getOp("emb_1");
  auto emb = std::dynamic_pointer_cast<Embedding>(emb_op);

  if (!emb) {
    throw std::runtime_error("Error casting 'emb_1' op to Embedding Op");
  }

  for (const auto& [k, v] : _tag_to_label) {
    _label_to_tag_map[v] = k;
  }
  _bolt_model =
      initializeBoltModel(fhr, emb->dim(), number_labels, emb->parameters());
  initializeNER(fhr, number_labels);
}

std::vector<std::vector<std::vector<std::pair<std::string, float>>>>
NerUDTModel::getTags(const std::vector<std::string>& sentences,
                     uint32_t top_k) const {
  return _classifier->getTags(sentences, top_k, _label_to_tag_map,
                              _tag_to_label);
}

metrics::History NerUDTModel::train(
    const dataset::DataSourcePtr& train_data, float learning_rate,
    uint32_t epochs, size_t batch_size,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics) {
  return _classifier->train(train_data, learning_rate, epochs, batch_size,
                            train_metrics, val_data, val_metrics);
}

ar::ConstArchivePtr NerUDTModel::toArchive() const {
  auto map = ar::Map::make();

  map->set("bolt_model", _bolt_model->toArchive(/*with_optimizer*/ false));

  map->set("tokens_column", ar::str(_tokens_column));
  map->set("tags_column", ar::str(_tags_column));

  auto tokenizers = ar::List::make();
  for (const auto& t : _target_word_tokenizers) {
    tokenizers->append(t->toArchive());
  }
  map->set("target_word_tokenizers", tokenizers);

  ar::MapStrU64 tag_to_label;
  for (const auto& [label, tag] : _tag_to_label) {
    tag_to_label[label] = tag;
  }
  map->set("tag_to_label", ar::mapStrU64(tag_to_label));

  if (_feature_enhancement_config.has_value()) {
    map->set("feature_enhancement_config",
             _feature_enhancement_config->toArchive());
  }

  return map;
}

std::shared_ptr<NerUDTModel> NerUDTModel::fromArchive(
    const ar::Archive& archive) {
  bolt::ModelPtr bolt_model =
      bolt::Model::fromArchive(*archive.get("bolt_model"));

  std::string tokens_column = archive.getAs<std::string>("tokens_column");
  std::string tags_column = archive.getAs<std::string>("tags_column");

  std::vector<dataset::TextTokenizerPtr> target_word_tokenizers;
  for (const auto& t : archive.get("target_word_tokenizers")->list()) {
    target_word_tokenizers.push_back(dataset::TextTokenizer::fromArchive(*t));
  }

  std::unordered_map<std::string, uint32_t> tag_to_label;
  for (const auto& [k, v] : archive.getAs<ar::MapStrU64>("tag_to_label")) {
    tag_to_label[k] = v;
  }

  std::optional<data::FeatureEnhancementConfig> feature_enhancement_config =
      std::nullopt;

  if (archive.contains("feature_enhancement_config")) {
    feature_enhancement_config = data::FeatureEnhancementConfig(
        *archive.get("feature_enhancement_config"));
  }

  return std::make_shared<NerUDTModel>(
      NerUDTModel(bolt_model, tokens_column, tags_column, tag_to_label,
                  target_word_tokenizers, feature_enhancement_config));
}

}  // namespace thirdai::bolt::NER