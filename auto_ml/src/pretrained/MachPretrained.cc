#include "MachPretrained.h"
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/NextWordPrediction.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/TextTokenizer.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/mach/MachIndex.h>

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

MachPretrained::MachPretrained(std::string input_column,
                               dataset::TextTokenizerPtr tokenizer,
                               size_t vocab_size, size_t emb_dim,
                               size_t output_dim, size_t n_models)
    : _input_column(std::move(input_column)) {
  for (size_t i = 0; i < n_models; i++) {
    _indexes.push_back(dataset::mach::MachIndex::make(
        output_dim, /*num_hashes=*/1, vocab_size, /*seed=*/i + 1));

    _models.push_back(buildModel(vocab_size, emb_dim, output_dim));
  }

  _tokenizer = std::make_shared<data::TextTokenizer>(
      _input_column, _input_column, std::nullopt, std::move(tokenizer),
      dataset::NGramEncoder::make(1), false, vocab_size);

  _nwp = std::make_shared<data::NextWordPrediction>(
      _input_column, _source_column, _target_column);
}

std::vector<bolt::metrics::History> MachPretrained::train(
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

std::vector<std::vector<uint32_t>> MachPretrained::decodeHashes(
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

}  // namespace thirdai::automl