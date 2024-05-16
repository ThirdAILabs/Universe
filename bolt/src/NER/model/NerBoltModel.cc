#include "NerBoltModel.h"
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
#include <optional>
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
NerBoltModel::NerBoltModel(
    std::shared_ptr<NerBoltModel>& pretrained_model,
    std::unordered_map<std::string, uint32_t> tag_to_label,
    std::string token_column, std::string tag_column)
    : _tag_to_label(tag_to_label),
      _source_column(std::move(token_column)),
      _target_column(std::move(tag_column)) {
  auto maxPair = std::max_element(
      tag_to_label.begin(), tag_to_label.end(),
      [](const auto& a, const auto& b) { return a.second < b.second; });
  auto num_labels = maxPair->second + 1;

  auto ops = pretrained_model->getBoltModel()->ops();
  bool found = std::any_of(ops.begin(), ops.end(), [](const bolt::OpPtr& op) {
    return op->name() == "emb_1";
  });

  if (!found) {
    throw std::runtime_error(
        "Error: No operation named 'emb_1' found in Pretrained Model");
  }
  auto emb = std::dynamic_pointer_cast<Embedding>(
      pretrained_model->getBoltModel()->getOp("emb_1"));

  if (!emb) {
    throw std::runtime_error("Error casting 'emb_1' op to Embedding Op");
  }

  _vocab_size = emb->inputDim();

  auto emb_weights = emb->parameters();

  auto inputs = std::vector<bolt::ComputationPtr>(
      {bolt::Input::make(_vocab_size), bolt::Input::make(_vocab_size),
       bolt::Input::make(_vocab_size)});

  auto emb_op = bolt::Embedding::make(emb->dim(), _vocab_size, "relu",
                                      /* bias= */ false);
  auto* pretrained_weights = emb_weights[0];

  emb_op->setEmbeddings(pretrained_weights->data());

  auto tokens_embedding = emb_op->apply(inputs[0]);
  auto token_front_embedding = emb_op->apply(inputs[1]);
  auto token_behind_embedding = emb_op->apply(inputs[2]);

  auto concat =
      bolt::Concatenate::make()->apply(std::vector<bolt::ComputationPtr>(
          {token_front_embedding, token_behind_embedding}));

  auto weighted_sum = bolt::WeightedSum::make(2, emb->dim())->apply(concat);

  concat = bolt::Concatenate::make()->apply(
      std::vector<bolt::ComputationPtr>({tokens_embedding, weighted_sum}));

  auto output =
      bolt::FullyConnected::make(num_labels, concat->dim(), 1, "softmax",
                                 /* sampling= */ nullptr, /* use_bias= */ true)
          ->apply(concat);

  auto labels = bolt::Input::make(num_labels);
  auto loss = bolt::CategoricalCrossEntropy::make(output, labels);

  _bolt_model = bolt::Model::make({inputs}, {output}, {loss});

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

std::vector<PerTokenListPredictions> NerBoltModel::getTags(
    std::vector<std::vector<std::string>> tokens, uint32_t top_k) {
  return thirdai::bolt::getTags(tokens, top_k, _source_column,
                                _inference_transforms, _bolt_inputs,
                                _bolt_model);
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