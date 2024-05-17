#include "NerUDTModel.h"
#include <bolt/src/NER/model/NER.h>
#include <bolt/src/NER/model/utils.h>
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
#include <data/src/transformations/ner/NerTokenizationUnigram.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt {

void NerUDTModel::initializeNER() {
  auto train_transformation = thirdai::data::NerTokenizerUnigram(
      /*tokens_column=*/_tokens_column,
      /*featurized_sentence_column=*/_featurized_sentence_column,
      /*target_column=*/_tags_column, /*target_dim=*/_number_labels,
      /*fhr_dim=*/_fhr, /*dyadic_num_intervals=*/_dyadic_num_intervals,
      /*target_word_tokenizers=*/_target_word_tokenizers,
      /*tag_to_label=*/_tag_to_label);

  auto inference_transformation = thirdai::data::NerTokenizerUnigram(
      /*tokens_column=*/_tokens_column,
      /*featurized_sentence_column=*/_featurized_sentence_column,
      /*target_column=*/std::nullopt, /*target_dim=*/std::nullopt,
      /*fhr_dim=*/_fhr, /*dyadic_num_intervals=*/_dyadic_num_intervals,
      /*target_word_tokenizers=*/_target_word_tokenizers,
      /*tag_to_label=*/_tag_to_label);

  _train_transforms = data::Pipeline::make(
      {std::make_shared<thirdai::data::NerTokenizerUnigram>(
          train_transformation)});

  _inference_transforms = data::Pipeline::make(
      {std::make_shared<thirdai::data::NerTokenizerUnigram>(
          inference_transformation)});

  _bolt_inputs = {
      data::OutputColumns(train_transformation.getFeaturizedIndicesColumn())};
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
    std::vector<dataset::TextTokenizerPtr> target_word_tokenizers)
    : _bolt_model(std::move(model)),
      _tokens_column(std::move(tokens_column)),
      _tags_column(std::move(tags_column)),
      _target_word_tokenizers(std::move(target_word_tokenizers)),
      _tag_to_label(std::move(tag_to_label)) {
  auto input_dims = _bolt_model->inputDims();
  if (input_dims.size() != 1) {
    throw std::logic_error(
        "Can only train a bolt model with a Single Input. Found model with "
        "number of inputs: " +
        std::to_string(input_dims.size()));
  }

  _fhr = input_dims[0];
  auto maxPair = std::max_element(
      _tag_to_label.begin(), _tag_to_label.end(),
      [](const auto& a, const auto& b) { return a.second < b.second; });
  _number_labels = maxPair->second + 1;

  initializeNER();
}

NerUDTModel::NerUDTModel(
    std::string tokens_column, std::string tags_column,
    std::unordered_map<std::string, uint32_t> tag_to_label,
    std::vector<dataset::TextTokenizerPtr> target_word_tokenizers)
    : _tokens_column(std::move(tokens_column)),
      _tags_column(std::move(tags_column)),
      _target_word_tokenizers(std::move(target_word_tokenizers)),
      _tag_to_label(tag_to_label),
      _fhr(100000) {
  auto maxPair = std::max_element(
      tag_to_label.begin(), tag_to_label.end(),
      [](const auto& a, const auto& b) { return a.second < b.second; });
  _number_labels = maxPair->second + 1;

  _bolt_model = initializeBoltModel(_fhr, 2000, _number_labels);

  initializeNER();
}

NerUDTModel::NerUDTModel(std::shared_ptr<NerUDTModel>& pretrained_model,
                         std::string tokens_column, std::string tags_column,
                         std::unordered_map<std::string, uint32_t> tag_to_label)
    : _tokens_column(std::move(tokens_column)),
      _tags_column(std::move(tags_column)),
      _target_word_tokenizers(pretrained_model->getTargetWordTokenizers()),
      _tag_to_label(tag_to_label),
      _fhr(pretrained_model->getBoltModel()->inputDims()[0]) {
  auto maxPair = std::max_element(
      tag_to_label.begin(), tag_to_label.end(),
      [](const auto& a, const auto& b) { return a.second < b.second; });
  _number_labels = maxPair->second + 1;

  auto emb_op = 
      pretrained_model->getBoltModel()->getOp("emb_1");
  auto emb = std::dynamic_pointer_cast<Embedding>(emb_op);

  if (!emb) {
    throw std::runtime_error("Error casting 'emb_1' op to Embedding Op");
  }
  _bolt_model =
      initializeBoltModel(_fhr, emb->dim(), _number_labels, emb->parameters());
  initializeNER();
}

std::vector<PerTokenListPredictions> NerUDTModel::getTags(
    std::vector<std::vector<std::string>> tokens, uint32_t top_k) {
  return thirdai::bolt::getTags(tokens, top_k, _tokens_column,
                                _inference_transforms, _bolt_inputs,
                                _bolt_model);
}

metrics::History NerUDTModel::train(
    const dataset::DataSourcePtr& train_data, float learning_rate,
    uint32_t epochs, size_t batch_size,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics) {
  auto train_dataset =
      getDataLoader(train_data, batch_size, /* shuffle= */ true).all();
  std::optional<bolt::LabeledDataset> val_dataset = std::nullopt;
  if (val_data) {
    val_dataset =
        getDataLoader(val_data, batch_size, /* shuffle= */ false).all();
  }

  auto train_data_input = train_dataset.first;
  auto train_data_label = train_dataset.second;

  Trainer trainer(_bolt_model);

  trainer.train_with_metric_names(
      train_dataset, learning_rate, epochs, train_metrics, val_dataset,
      val_metrics, /* steps_per_validation= */ std::nullopt,
      /* use_sparsity_in_validation= */ false, /* callbacks= */ {},
      /* autotune_rehash_rebuild= */ false, /* verbose= */ true);
  return trainer.getHistory();
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
  return std::make_shared<NerUDTModel>(NerUDTModel(bolt_model, tokens_column,
                                                   tags_column, tag_to_label,
                                                   target_word_tokenizers));
}

void NerUDTModel::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void NerUDTModel::save_stream(std::ostream& output) const {
  ar::serialize(toArchive(), output);
}

std::shared_ptr<NerUDTModel> NerUDTModel::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

std::shared_ptr<NerUDTModel> NerUDTModel::load_stream(std::istream& input) {
  auto archive = ar::deserialize(input);
  return fromArchive(*archive);
}

data::Loader NerUDTModel::getDataLoader(const dataset::DataSourcePtr& data,
                                        size_t batch_size, bool shuffle) {
  auto data_iter =
      data::JsonIterator::make(data, {_tokens_column, _tags_column}, 1000);
  return data::Loader(data_iter, _train_transforms, nullptr, _bolt_inputs,
                      {data::OutputColumns(_tags_column)},
                      /* batch_size= */ batch_size,
                      /* shuffle= */ shuffle, /* verbose= */ true,
                      /* shuffle_buffer_size= */ 20000);
}
}  // namespace thirdai::bolt