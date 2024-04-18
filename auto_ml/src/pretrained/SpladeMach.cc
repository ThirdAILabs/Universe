#include "SpladeMach.h"
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/NextWordPrediction.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/TextTokenizer.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/mach/MachIndex.h>
#include <utility>

namespace thirdai::automl {

bolt::ModelPtr buildModel(size_t vocab_size, size_t emb_dim,
                          size_t output_dim) {
  auto input = bolt::Input::make(vocab_size);

  auto emb = bolt::Embedding::make(emb_dim, vocab_size, "relu")->apply(input);

  auto out = bolt::FullyConnected::make(output_dim, emb_dim, 0.02, "sigmoid")
                 ->apply(emb);

  auto loss =
      bolt::BinaryCrossEntropy::make(out, bolt::Input::make(output_dim));

  return bolt::Model::make({input}, {out}, {loss});
}

SpladeMach::SpladeMach(std::string input_column,
                       dataset::TextTokenizerPtr tokenizer, size_t vocab_size,
                       size_t emb_dim, size_t output_dim, size_t n_models)
    : _input_column(std::move(input_column)), _vocab_size(vocab_size) {
  for (size_t i = 0; i < n_models; i++) {
    _indexes.push_back(dataset::mach::MachIndex::make(
        output_dim, /*num_hashes=*/1, vocab_size, /*seed=*/i + 1));

    _models.push_back(buildModel(vocab_size, emb_dim, output_dim));
  }

  _tokenizer = std::make_shared<data::TextTokenizer>(
      _input_column, _input_column, std::nullopt, std::move(tokenizer),
      dataset::NGramEncoder::make(1), false, vocab_size);
}

SpladeMach::SpladeMach(std::string input_column,
                       std::vector<bolt::ModelPtr> models,
                       std::vector<data::MachIndexPtr> indexes,
                       dataset::TextTokenizerPtr tokenizer, size_t vocab_size)
    : _models(std::move(models)),
      _indexes(std::move(indexes)),
      _input_column(std::move(input_column)),
      _vocab_size(vocab_size) {
  _tokenizer = std::make_shared<data::TextTokenizer>(
      _input_column, _input_column, std::nullopt, std::move(tokenizer),
      dataset::NGramEncoder::make(1), false, vocab_size);
}

std::vector<bolt::metrics::History> SpladeMach::train(
    const data::ColumnMapIteratorPtr& train_data, size_t epochs,
    size_t batch_size, float learning_rate,
    const data::ColumnMapIteratorPtr& val_data) {
  std::vector<bolt::metrics::History> histories;

  for (size_t i = 0; i < _models.size(); i++) {
    bolt::Trainer trainer(_models[i]);

    auto state = data::State::make(_indexes.at(i), nullptr);
    auto train_loader = getDataLoader(train_data, state, batch_size);

    data::LoaderPtr val_loader = nullptr;
    if (val_data) {
      val_loader = getDataLoader(val_data, state, batch_size);
    }

    auto metrics = trainer.train_with_data_loader(
        train_loader, learning_rate, epochs,
        /*max_in_memory_batches=*/100, {}, val_loader,
        {{"loss", std::make_shared<bolt::metrics::LossMetric>(
                      _models[0]->losses().at(0))}});

    histories.push_back(metrics);

    train_data->restart();
    if (val_data) {
      val_data->restart();
    }
  }

  return histories;
}

std::vector<std::vector<uint32_t>> SpladeMach::getTopHashBuckets(
    std::vector<std::string> phrases, size_t hashes_per_model) {
  data::ColumnMap columns({{_input_column, data::ValueColumn<std::string>::make(
                                               std::move(phrases))}});

  columns = _tokenizer->applyStateless(columns);

  auto tensor = data::toTensors(columns, {data::OutputColumns(_source_column)});

  std::vector<std::vector<uint32_t>> hashes(tensor[0]->batchSize());

  size_t offset = 0;
  for (const auto& model : _models) {
    auto output = model->forward(tensor).at(0);

    for (size_t i = 0; i < output->batchSize(); i++) {
      auto topk = output->getVector(i).topKNeurons(hashes_per_model);

      while (!topk.empty()) {
        hashes[i].push_back(offset + topk.top().second);
        topk.pop();
      }
    }

    offset += model->outputs().at(0)->dim();
  }

  return hashes;
}

std::vector<uint32_t> SpladeMach::getTopTokens(std::string phrase,
                                               size_t num_tokens,
                                               size_t num_buckets_to_decode) {
  data::ColumnMap columns({{_input_column, data::ValueColumn<std::string>::make(
                                               {std::move(phrase)})}});

  columns = _tokenizer->applyStateless(columns);

  auto tensor = data::toTensors(columns, {data::OutputColumns(_source_column)});

  std::vector<TopKActivationsQueue> top_k_buckets;

  for (const auto& model : _models) {
    auto output = model->forward(tensor).at(0);
    top_k_buckets.push_back(
        output->getVector(0).topKNeurons(num_buckets_to_decode));
  }

  std::unordered_map<uint32_t, float> score_map;
  for (size_t model_id = 0; model_id < _models.size(); model_id++) {
    auto model_top_k_buckets = top_k_buckets[model_id];
    while (!model_top_k_buckets.empty()) {
      auto bucket_id = model_top_k_buckets.top().second;
      auto bucket_activation = model_top_k_buckets.top().first;

      model_top_k_buckets.pop();
      auto indices = _indexes[model_id]->getEntities(bucket_id);

      for (const auto& label : indices) {
        if (!score_map.count(label)) {
          score_map[label] = bucket_activation;
        } else {
          score_map[label] += bucket_activation;
        }
      }
    }
  }

  std::vector<std::pair<uint32_t, float>> score_map_v(score_map.begin(),
                                                      score_map.end());

  std::sort(score_map_v.begin(), score_map_v.end(),
            [](const auto& a, const auto& b) {
              return a.second > b.second;  // Sort by value descending
            });

  std::vector<uint32_t> top_tokens;
  for (const auto& top_token_p : score_map_v) {
    top_tokens.push_back(top_token_p.first);
    if (top_tokens.size() == num_tokens) {
      break;
    }
  }

  return top_tokens;
}

ar::ConstArchivePtr SpladeMach::toArchive() const {
  auto mach_pretrained = ar::Map::make();

  auto models = ar::List::make();
  for (const auto& model : _models) {
    models->append(model->toArchive(/*with_optimizer*/ false));
  }
  mach_pretrained->set("models", models);

  auto indexes = ar::List::make();
  for (const auto& index : _indexes) {
    indexes->append(index->toArchive());
  }
  mach_pretrained->set("indexes", indexes);

  mach_pretrained->set("tokenizer", _tokenizer->getTokenizer()->toArchive());

  mach_pretrained->set("input_column", ar::str(_input_column));

  mach_pretrained->set("vocab_size", ar::u64(_vocab_size));

  return mach_pretrained;
}

std::shared_ptr<SpladeMach> SpladeMach::fromArchive(
    const ar::Archive& archive) {
  std::vector<bolt::ModelPtr> models;
  for (const auto& model : archive.get("models")->list()) {
    models.push_back(bolt::Model::fromArchive(*model));
  }

  std::vector<data::MachIndexPtr> indexes;
  for (const auto& index : archive.get("indexes")->list()) {
    indexes.push_back(dataset::mach::MachIndex::fromArchive(*index));
  }

  dataset::TextTokenizerPtr tokenizer =
      dataset::TextTokenizer::fromArchive(*archive.get("tokenizer"));

  std::string input_column = archive.str("input_column");

  auto vocab_size = archive.u64("vocab_size");

  return std::make_shared<SpladeMach>(
      SpladeMach(input_column, models, indexes, tokenizer, vocab_size));
}

void SpladeMach::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void SpladeMach::save_stream(std::ostream& output) const {
  ar::serialize(toArchive(), output);
}

std::shared_ptr<SpladeMach> SpladeMach::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

std::shared_ptr<SpladeMach> SpladeMach::load_stream(std::istream& input) {
  auto archive = ar::deserialize(input);
  return fromArchive(*archive);
}

}  // namespace thirdai::automl