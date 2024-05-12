#include "NerBoltModel.h"
#include <cereal/archives/binary.hpp>
#include <bolt/src/NER/model/NER.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt_vector/src/BoltVector.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/NerTokenFromStringArray.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <cmath>
#include <optional>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::bolt {
NerBoltModel::NerBoltModel(
    bolt::ModelPtr model,
    std::unordered_map<std::string, uint32_t> tag_to_label)
    : _bolt_model(std::move(model)), _tag_to_label(std::move(tag_to_label)) {
  _train_transforms = getTransformations(true);
  _inference_transforms = getTransformations(false);
  _bolt_inputs = {data::OutputColumns("tokens"),
                  data::OutputColumns("token_front"),
                  data::OutputColumns("token_behind")};
}

data::PipelinePtr NerBoltModel::getTransformations(bool inference) {
  data::PipelinePtr transform;
  if (!inference) {
    transform =
        data::Pipeline::make({std::make_shared<data::NerTokenFromStringArray>(
            _source_column, "tokens", "token_front", "token_behind",
            std::nullopt, std::nullopt)});
  } else {
    transform =
        data::Pipeline::make({std::make_shared<data::NerTokenFromStringArray>(
            _source_column, "tokens", "token_front", "token_behind",
            _target_column, _tag_to_label)});
  }
  transform = transform->then(std::make_shared<data::StringToTokenArray>(
      "tokens", "tokens", ' ', _vocab_size));
  transform = transform->then(std::make_shared<data::StringToTokenArray>(
      "token_front", "token_front", ' ', _vocab_size));
  transform = transform->then(std::make_shared<data::StringToTokenArray>(
      "token_behind", "token_behind", ' ', _vocab_size));
  return transform;
}

data::Loader NerBoltModel::getDataLoader(const dataset::DataSourcePtr& data,
                                         size_t batch_size, bool shuffle) {
  auto data_iter =
      data::JsonIterator::make(data, {_source_column, _target_column}, 1000);
  return data::Loader(data_iter, _train_transforms, nullptr, _bolt_inputs,
                      {data::OutputColumns(_target_column)},
                      /* batch_size= */ batch_size,
                      /* shuffle= */ shuffle, /* verbose= */ true,
                      /* shuffle_buffer_size= */ 20000);
}
metrics::History NerBoltModel::train(
    const dataset::DataSourcePtr& train_data, float learning_rate,
    uint32_t epochs, size_t batch_size,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics) {
  auto train_dataset =
      getDataLoader(train_data, batch_size, /* shuffle= */ true).all();
  auto val_dataset =
      getDataLoader(val_data, batch_size, /* shuffle= */ false).all();

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

std::vector<PerTokenListPredictions> NerBoltModel::getTags(
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
      {{_source_column, data::ArrayColumn<std::string>::make(std::move(tokens),
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

ar::ConstArchivePtr NerBoltModel::toArchive() const {
  auto ner_bolt_model = ar::Map::make();

  ner_bolt_model->set("bolt_model",
                      _bolt_model->toArchive(/*with_optimizer*/ false));

  ar::MapStrU64 tag_to_label;
  for (const auto& [label, tag] : _tag_to_label) {
    tag_to_label[label] = tag;
  }
  ner_bolt_model->set("tag_to_label", ar::mapStrU64(tag_to_label));

  return ner_bolt_model;
}

std::shared_ptr<NerBoltModel> NerBoltModel::fromArchive(
    const ar::Archive& archive) {
  bolt::ModelPtr bolt_model =
      bolt::Model::fromArchive(*archive.get("bolt_model"));
  std::unordered_map<std::string, uint32_t> tag_to_label;
  for (const auto& [k, v] : archive.getAs<ar::MapStrU64>("tag_to_label")) {
    tag_to_label[k] = v;
  }
  return std::make_shared<NerBoltModel>(NerBoltModel(bolt_model, tag_to_label));
}

void NerBoltModel::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void NerBoltModel::save_stream(std::ostream& output) const {
  ar::serialize(toArchive(), output);
}

std::shared_ptr<NerBoltModel> NerBoltModel::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

std::shared_ptr<NerBoltModel> NerBoltModel::load_stream(std::istream& input) {
  auto archive = ar::deserialize(input);
  return fromArchive(*archive);
}
}  // namespace thirdai::bolt