#include "NerPretrainedModel.h"
#include <cereal/archives/binary.hpp>
#include <bolt/src/NER/model/NER.h>
#include <bolt/src/NER/model/utils.h>
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
#include <data/src/transformations/NerTokenFromStringArray.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::bolt {

bolt::ModelPtr NerPretrainedModel::getBoltModel(
    std::string& pretrained_model_path,
    std::unordered_map<std::string, uint32_t> tag_to_label,
    uint32_t vocab_size) {
  auto pretrained_model = bolt::Model::load(pretrained_model_path);
  if (pretrained_model->inputs()[0]->dim() != 50257) {
    throw std::invalid_argument(
        "Model input should have same vocab as GPT2Tokenizer");
  }
  uint32_t num_labels = getMaxLabelFromTagToLabel(std::move(tag_to_label));

  auto ops = pretrained_model->ops();
  bool found = std::any_of(ops.begin(), ops.end(), [](const bolt::OpPtr& op) {
    return op->name() == "emb_1";
  });

  if (!found) {
    throw std::runtime_error(
        "Error: No operation named 'emb_1' found in Pretrained Model");
  }
  auto emb =
      std::dynamic_pointer_cast<Embedding>(pretrained_model->getOp("emb_1"));

  auto emb_weights = emb->parameters();

  auto inputs = std::vector<bolt::ComputationPtr>(
      {bolt::Input::make(vocab_size), bolt::Input::make(vocab_size),
       bolt::Input::make(vocab_size)});

  auto emb_op = bolt::Embedding::make(6000, vocab_size, "relu",
                                      /* bias= */ false);
  auto* pretrained_weights = emb_weights[0];

  if (pretrained_weights->size() == 6000 * vocab_size) {
    emb_op->setEmbeddings(pretrained_weights->data());
  } else {
    throw std::runtime_error("Size mismatch in embeddings vector.");
  }
  auto tokens_embedding = emb_op->apply(inputs[0]);
  auto token_front_embedding = emb_op->apply(inputs[1]);
  auto token_behind_embedding = emb_op->apply(inputs[2]);

  auto concat =
      bolt::Concatenate::make()->apply(std::vector<bolt::ComputationPtr>(
          {token_front_embedding, token_behind_embedding}));

  auto weighted_sum = bolt::WeightedSum::make(2, 6000)->apply(concat);

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

data::PipelinePtr NerPretrainedModel::getTransformations(bool inference) {
  data::PipelinePtr transform;
  if (inference) {
    transform =
        data::Pipeline::make({std::make_shared<data::NerTokenFromStringArray>(
            _tokens_column, "tokens", "token_front", "token_behind",
            std::nullopt, std::nullopt)});
  } else {
    transform =
        data::Pipeline::make({std::make_shared<data::NerTokenFromStringArray>(
            _tokens_column, "tokens", "token_front", "token_behind",
            _tags_column, _tag_to_label)});
  }
  transform = transform->then(std::make_shared<data::StringToTokenArray>(
      "tokens", "tokens", ' ', _vocab_size));
  transform = transform->then(std::make_shared<data::StringToTokenArray>(
      "token_front", "token_front", ' ', _vocab_size));
  transform = transform->then(std::make_shared<data::StringToTokenArray>(
      "token_behind", "token_behind", ' ', _vocab_size));
  return transform;
}

NerPretrainedModel::NerPretrainedModel(
    bolt::ModelPtr model, std::string tokens_column, std::string tags_column,
    std::unordered_map<std::string, uint32_t> tag_to_label)
    : _bolt_model(std::move(model)),
      _tokens_column(std::move(tokens_column)),
      _tags_column(std::move(tags_column)),
      _tag_to_label(std::move(tag_to_label)) {
  auto train_transforms = getTransformations(/*inference=*/false);
  auto inference_transforms = getTransformations(/*inference=*/true);
  auto bolt_inputs = {data::OutputColumns("tokens"),
                      data::OutputColumns("token_front"),
                      data::OutputColumns("token_behind")};
  _classifier = std::make_shared<NerClassifier>(
      _bolt_model, bolt_inputs, train_transforms, inference_transforms,
      _tokens_column, _tags_column);
}

NerPretrainedModel::NerPretrainedModel(
    std::string& pretrained_model_path, std::string tokens_column,
    std::string tags_column,
    std::unordered_map<std::string, uint32_t> tag_to_label)
    : _tokens_column(std::move(tokens_column)),
      _tags_column(std::move(tags_column)),
      _tag_to_label(std::move(tag_to_label)) {
  _bolt_model = getBoltModel(pretrained_model_path, tag_to_label, 50257);
  auto train_transforms = getTransformations(/*inference=*/false);
  auto inference_transforms = getTransformations(/*inference=*/true);
  auto bolt_inputs = {data::OutputColumns("tokens"),
                      data::OutputColumns("token_front"),
                      data::OutputColumns("token_behind")};
  _classifier = std::make_shared<NerClassifier>(
      _bolt_model, bolt_inputs, train_transforms, inference_transforms,
      _tokens_column, _tags_column);
}

metrics::History NerPretrainedModel::train(
    const dataset::DataSourcePtr& train_data, float learning_rate,
    uint32_t epochs, size_t batch_size,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics) const {
  return _classifier->train(train_data, learning_rate, epochs, batch_size,
                            train_metrics, val_data, val_metrics);
}

std::vector<PerTokenListPredictions> NerPretrainedModel::getTags(
    std::vector<std::vector<std::string>> tokens, uint32_t top_k) const {
  return _classifier->getTags(tokens, top_k);
}

ar::ConstArchivePtr NerPretrainedModel::toArchive() const {
  auto map = ar::Map::make();

  map->set("bolt_model", _bolt_model->toArchive(/*with_optimizer*/ false));

  map->set("tokens_column", ar::str(_tokens_column));
  map->set("tags_column", ar::str(_tags_column));

  ar::MapStrU64 tag_to_label;
  for (const auto& [label, tag] : _tag_to_label) {
    tag_to_label[label] = tag;
  }
  map->set("tag_to_label", ar::mapStrU64(tag_to_label));

  return map;
}

std::shared_ptr<NerPretrainedModel> NerPretrainedModel::fromArchive(
    const ar::Archive& archive) {
  bolt::ModelPtr bolt_model =
      bolt::Model::fromArchive(*archive.get("bolt_model"));

  std::string tokens_column = archive.getAs<std::string>("tokens_column");
  std::string tags_column = archive.getAs<std::string>("tags_column");

  std::unordered_map<std::string, uint32_t> tag_to_label;
  for (const auto& [k, v] : archive.getAs<ar::MapStrU64>("tag_to_label")) {
    tag_to_label[k] = v;
  }
  return std::make_shared<NerPretrainedModel>(
      NerPretrainedModel(bolt_model, tokens_column, tags_column, tag_to_label));
}

void NerPretrainedModel::save_stream(std::ostream& output) const {
  ar::serialize(toArchive(), output);
}

std::shared_ptr<NerPretrainedModel> NerPretrainedModel::load_stream(
    std::istream& input) {
  auto archive = ar::deserialize(input);
  return fromArchive(*archive);
}
}  // namespace thirdai::bolt