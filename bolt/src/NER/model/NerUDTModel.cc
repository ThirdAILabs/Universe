#include "NerUDTModel.h"
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
#include <data/src/transformations/ner/NerTokenizationUnigram.h>
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt {

void NerUDTModel::initialize() {
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

  initialize();
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

  auto input = bolt::Input::make(_fhr);

  auto hidden = bolt::Embedding::make(2000, 100000, "relu",
                                      /* bias= */ true)
                    ->apply(input);

  auto output =
      bolt::FullyConnected::make(_number_labels, hidden->dim(), 1, "softmax",
                                 /* sampling= */ nullptr, /* use_bias= */ true)
          ->apply(hidden);

  auto labels = bolt::Input::make(_number_labels);
  auto loss = bolt::CategoricalCrossEntropy::make(output, labels);

  _bolt_model = bolt::Model::make({input}, {output}, {loss});
  initialize();
}

std::vector<PerTokenListPredictions> NerUDTModel::getTags(
    std::vector<std::vector<std::string>> tokens, uint32_t top_k) {
  std::vector<PerTokenListPredictions> tags_and_scores;
  tags_and_scores.reserve(tokens.size());

  for (const auto& sub_vector : tokens) {
    PerTokenListPredictions predictions;
    predictions.reserve(sub_vector.size());
    for (size_t i = 0; i < sub_vector.size(); i++) {
      predictions.push_back(PerTokenPredictions());
    }
    tags_and_scores.push_back(predictions);
  }
  data::ColumnMap data(data::ColumnMap(
      {{_tokens_column, data::ArrayColumn<std::string>::make(std::move(tokens),
                                                             std::nullopt)}}));

  auto columns = _inference_transforms->applyStateless(data);
  auto tensors = data::toTensorBatches(columns, _bolt_inputs, 2048);

  size_t sub_vector_index = 0;
  size_t token_index = 0;

  for (const auto& batch : tensors) {
    auto outputs = _bolt_model->forward(batch).at(0);

    for (size_t i = 0; i < outputs->batchSize(); i += 1) {
      if (token_index >= tags_and_scores[sub_vector_index].size()) {
        token_index = 0;
        sub_vector_index++;
      }
      auto token_level_predictions = outputs->getVector(i).topKNeurons(top_k);
      while (!token_level_predictions.empty()) {
        float score = token_level_predictions.top().first;
        uint32_t tag = token_level_predictions.top().second;
        tags_and_scores[sub_vector_index][token_index].push_back({tag, score});
        token_level_predictions.pop();
      }
      if (sub_vector_index >= tags_and_scores.size()) {
        throw std::runtime_error("tags indices not matching");
      }
      token_index += 1;
    }
  }
  return tags_and_scores;
}

metrics::History NerUDTModel::train(
    const dataset::DataSourcePtr& train_data, float learning_rate,
    uint32_t epochs, size_t batch_size,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics) {
  auto train_dataset =
      getDataLoader(train_data, batch_size, /* shuffle= */ true).all();
  bolt::LabeledDataset val_dataset;
  if (val_data) {
    val_dataset =
        getDataLoader(val_data, batch_size, /* shuffle= */ false).all();
  }

  auto train_data_input = train_dataset.first;
  auto train_data_label = train_dataset.second;

  Trainer trainer(_bolt_model);

  // We cannot use train_with_dataset_loader, since it is using the older
  // dataset::DatasetLoader while dyadic model is using data::Loader
  for (uint32_t e = 0; e < epochs; e++) {
    trainer.train_with_metric_names(
        train_dataset, learning_rate, 1, train_metrics, val_dataset,
        val_metrics, /* steps_per_validation= */ std::nullopt,
        /* use_sparsity_in_validation= */ false, /* callbacks= */ {},
        /* autotune_rehash_rebuild= */ false, /* verbose= */ true);
  }
  return trainer.getHistory();
}

ar::ConstArchivePtr NerUDTModel::toArchive() const {
  auto map = ar::Map::make();

  map->set("bolt_model", _bolt_model->toArchive(/*with_optimizer*/ false));

  map->set("tokens_column", ar::str(_tokens_column));
  map->set("tags_column", ar::str(_tokens_column));

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