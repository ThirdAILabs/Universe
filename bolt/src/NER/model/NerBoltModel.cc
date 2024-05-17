#include "NerBoltModel.h"
#include <cereal/archives/binary.hpp>
#include <bolt/src/NER/model/NER.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/ops/WeightedSum.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt_vector/src/BoltVector.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/ner/NerTokenFromStringArray.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::bolt::NER {

bolt::ModelPtr NerBoltModel::initializeBoltModel(
    std::shared_ptr<NerBoltModel>& pretrained_model,
    std::unordered_map<std::string, uint32_t>& tag_to_label,
    uint32_t vocab_size) {
  auto num_labels = getMaxLabelFromTagToLabel(tag_to_label);

  auto emb_op_pretrained = pretrained_model->getBoltModel()->getOp("emb_1");
  auto emb = std::dynamic_pointer_cast<Embedding>(emb_op_pretrained);

  if (!emb) {
    throw std::runtime_error("Error casting 'emb_1' op to Embedding Op");
  }

  auto emb_weights = emb->parameters();

  auto inputs = std::vector<bolt::ComputationPtr>(
      {bolt::Input::make(vocab_size), bolt::Input::make(vocab_size),
       bolt::Input::make(vocab_size)});

  auto emb_op = bolt::Embedding::make(emb->dim(), vocab_size, "relu",
                                      /* bias= */ false);
  auto* pretrained_weights = emb_weights[0];

  emb_op->setEmbeddings(pretrained_weights->data());

  auto tokens_embedding = emb_op->apply(inputs[0]);
  auto token_next_embedding = emb_op->apply(inputs[1]);
  auto token_previous_embedding = emb_op->apply(inputs[2]);

  auto concat =
      bolt::Concatenate::make()->apply(std::vector<bolt::ComputationPtr>(
          {token_next_embedding, token_previous_embedding}));

  auto weighted_sum = bolt::WeightedSum::make(2, emb->dim())->apply(concat);

  concat = bolt::Concatenate::make()->apply(
      std::vector<bolt::ComputationPtr>({tokens_embedding, weighted_sum}));

  auto output =
      bolt::FullyConnected::make(num_labels, concat->dim(), 1, "softmax",
                                 /* sampling= */ nullptr, /* use_bias= */ true)
          ->apply(concat);

  auto labels = bolt::Input::make(num_labels);
  auto loss = bolt::CategoricalCrossEntropy::make(output, labels);

  return bolt::Model::make({inputs}, {output}, {loss});
}

NerBoltModel::NerBoltModel(
    bolt::ModelPtr model, std::string tokens_column, std::string tags_column,
    std::unordered_map<std::string, uint32_t> tag_to_label)
    : _bolt_model(std::move(model)),
      _tokens_column(std::move(tokens_column)),
      _tags_column(std::move(tags_column)),
      _tag_to_label(std::move(tag_to_label)) {
  auto train_transforms = getTransformations(/*inference=*/false);
  auto inference_transforms = getTransformations(/*inference=*/true);
  auto bolt_inputs = {data::OutputColumns("tokens"),
                      data::OutputColumns("token_next"),
                      data::OutputColumns("token_previous")};
  _classifier = std::make_shared<NerClassifier>(
      _bolt_model, bolt_inputs, train_transforms, inference_transforms,
      _tokens_column, _tags_column);

  for (const auto& [k, v] : _tag_to_label) {
    _label_to_tag_map[v] = k;
  }
}

NerBoltModel::NerBoltModel(
    std::shared_ptr<NerBoltModel>& pretrained_model, std::string tokens_column,
    std::string tags_column,
    std::unordered_map<std::string, uint32_t> tag_to_label)
    : _tokens_column(std::move(tokens_column)),
      _tags_column(std::move(tags_column)),
      _tag_to_label(std::move(tag_to_label)) {
  _bolt_model =
      initializeBoltModel(pretrained_model, _tag_to_label, _vocab_size);
  auto train_transforms = getTransformations(/*inference=*/false);
  auto inference_transforms = getTransformations(/*inference=*/true);
  auto bolt_inputs = {data::OutputColumns("tokens"),
                      data::OutputColumns("token_next"),
                      data::OutputColumns("token_previous")};
  _classifier = std::make_shared<NerClassifier>(
      _bolt_model, bolt_inputs, train_transforms, inference_transforms,
      _tokens_column, _tags_column);

  for (const auto& [k, v] : _tag_to_label) {
    _label_to_tag_map[v] = k;
  }
}

data::PipelinePtr NerBoltModel::getTransformations(bool inference) {
  data::PipelinePtr transform;
  if (inference) {
    transform =
        data::Pipeline::make({std::make_shared<data::NerTokenFromStringArray>(
            _tokens_column, "tokens", "token_next", "token_previous",
            std::nullopt, std::nullopt)});
  } else {
    transform =
        data::Pipeline::make({std::make_shared<data::NerTokenFromStringArray>(
            _tokens_column, "tokens", "token_next", "token_previous",
            _tags_column, _tag_to_label)});
  }
  transform = transform->then(std::make_shared<data::StringToTokenArray>(
      "tokens", "tokens", ' ', _vocab_size));
  transform = transform->then(std::make_shared<data::StringToTokenArray>(
      "token_next", "token_next", ' ', _vocab_size));
  transform = transform->then(std::make_shared<data::StringToTokenArray>(
      "token_previous", "token_previous", ' ', _vocab_size));
  return transform;
}

metrics::History NerBoltModel::train(
    const dataset::DataSourcePtr& train_data, float learning_rate,
    uint32_t epochs, size_t batch_size,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics) {
  return _classifier->train(train_data, learning_rate, epochs, batch_size,
                            train_metrics, val_data, val_metrics);
}

std::vector<std::vector<std::vector<std::pair<std::string, float>>>>
NerBoltModel::getTags(std::vector<std::vector<std::string>> tokens,
                      uint32_t top_k) const {
  return _classifier->getTags(tokens, top_k, _label_to_tag_map);
}

ar::ConstArchivePtr NerBoltModel::toArchive() const {
  auto ner_bolt_model = ar::Map::make();

  ner_bolt_model->set("bolt_model",
                      _bolt_model->toArchive(/*with_optimizer*/ false));

  ner_bolt_model->set("tokens_column", ar::str(_tokens_column));
  ner_bolt_model->set("tags_column", ar::str(_tags_column));

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

  std::string tokens_column = archive.getAs<std::string>("tokens_column");
  std::string tags_column = archive.getAs<std::string>("tags_column");

  std::unordered_map<std::string, uint32_t> tag_to_label;
  for (const auto& [k, v] : archive.getAs<ar::MapStrU64>("tag_to_label")) {
    tag_to_label[k] = v;
  }

  return std::make_shared<NerBoltModel>(
      NerBoltModel(bolt_model, /*tokens_column=*/tokens_column,
                   /*tags_column=*/tags_column, /*tag_to_label=*/tag_to_label));
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
}  // namespace thirdai::bolt::NER