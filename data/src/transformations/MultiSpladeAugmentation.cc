#include "MultiSpladeAugmentation.h"
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/utils/ProgressBar.h>
#include <bolt/src/utils/Timer.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/TextTokenizer.h>
#include <dataset/src/mach/MachIndex.h>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>

namespace thirdai::data {
MultiSpladeConfig::MultiSpladeConfig(
    const std::vector<std::string>& model_checkpoints,
    const std::vector<uint32_t>& mach_index_seeds,
    const std::string& tokenizer_vocab,
    std::optional<size_t> n_augmented_tokens,
    std::optional<float> augmentation_frac, bool filter_tokens,
    size_t batch_size, bool lowercase,
    std::optional<uint32_t> strong_sample_override)
    : tokenizer(std::make_shared<dataset::WordpieceTokenizer>(tokenizer_vocab,
                                                              lowercase)),
      n_augmented_tokens(n_augmented_tokens),
      augmentation_frac(augmentation_frac),
      filter_tokens(filter_tokens),
      batch_size(batch_size),
      strong_sample_override(strong_sample_override),
      _model_checkpoints(model_checkpoints),
      _tokenizer_vocab(tokenizer_vocab),
      _lowercase(lowercase) {
  for (const auto& checkpoint_path : model_checkpoints) {
    models.push_back(bolt::Model::load(checkpoint_path));
  }

  for (const auto& seed : mach_index_seeds) {
    dataset::mach::MachIndex index = dataset::mach::MachIndex(
        models[0]->outputs()[0]->dim(), 1, tokenizer->vocabSize(), seed);
    mach_indices.push_back(index);
    _mach_index_seeds.push_back(seed);
  }
}

MultiSpladeAugmentation::MultiSpladeAugmentation(
    std::string input_column, std::string output_column,
    const MultiSpladeConfig& config)
    : _input_column(std::move(input_column)),
      _output_column(std::move(output_column)),
      _models(config.models),
      _mach_indices(config.mach_indices),
      _tokenizer(config.tokenizer),
      _n_augmented_tokens(config.n_augmented_tokens),
      _augmentation_frac(config.augmentation_frac),
      _filter_tokens(config.filter_tokens),
      _batch_size(config.batch_size) {
  for (const auto& model : _models) {
    if (model->inputs().size() != 1 || model->outputs().size() != 1) {
      throw std::invalid_argument(
          "MultiSpladeAugmentation must have 1 input and output");
    }

    if (model->inputs()[0]->dim() != _tokenizer->vocabSize() ||
        model->outputs()[0]->dim() != _tokenizer->vocabSize()) {
      throw std::invalid_argument(
          "MultiSpladeAugmentation model input and output dim should match "
          "tokenizer "
          "vocab size.");
    }
  }
  if (_n_augmented_tokens.has_value() == _augmentation_frac.has_value()) {
    throw std::invalid_argument(
        "Must specified exactly one of n_augmented_tokens and "
        "augmentation_frac.");
  }
}

ColumnMap MultiSpladeAugmentation::apply(ColumnMap columns,
                                         State& state) const {
  (void)state;

  data::TextTokenizer tokenizer(
      /*input_column=*/_input_column, /*output_indices=*/_input_column,
      /*output_values=*/std::nullopt, /*tokenizer=*/_tokenizer,
      /*encoder=*/dataset::NGramEncoder::make(1), /*lowercase=*/false,
      /*dim=*/_tokenizer->vocabSize());
  auto tokenized_columns = tokenizer.applyStateless(columns);
  auto batches = data::toTensorBatches(
      tokenized_columns, {OutputColumns(_input_column)}, _batch_size);

  auto tokenized_text =
      tokenized_columns.getArrayColumn<uint32_t>(_input_column);
  std::vector<std::string> augmented_text(tokenized_text->numRows());

  ProgressBar bar("augmenting data", batches.size());
  bolt::utils::Timer timer;
  size_t row_index = 0;
  for (const auto& batch : batches) {
    std::vector<bolt::TensorPtr> model_outputs;
    for (const auto& model : _models) {
      auto output = model->forward(batch).at(0);
      model_outputs.push_back(output);
    }
#pragma omp parallel for default(none) \
    shared(tokenized_text, augmented_text, model_outputs, batch, row_index)
    for (size_t i = 0; i < batch.size(); i++) {
      std::vector<BoltVector> bolt_vectors;
      bolt_vectors.reserve(model_outputs.size());
      for (const auto& output : model_outputs) {
        bolt_vectors.push_back(output->getVector(i));
      }
      augmented_text[row_index + i] = decodeTopTokens(
          bolt_vectors, tokensToAdd(tokenized_text->row(row_index + i).size()));
    }
    row_index += batch.size();
    bar.increment();
  }
  timer.stop();

  bar.close("data augmentation completed in " +
            std::to_string(timer.seconds()) + "s.");
  ColumnMap output = columns;
  output.setColumn(_output_column,
                   ValueColumn<std::string>::make(std::move(augmented_text)));
  return output;
}

std::string MultiSpladeAugmentation::decodeTopTokens(
    const std::vector<BoltVector>& vectors, size_t k) const {
  std::string decoded;

  std::vector<TopKActivationsQueue> top_k_buckets;
  top_k_buckets.reserve(vectors.size());
  for (const auto& vec : vectors) {
    top_k_buckets.push_back(vec.topKNeurons(k));
  }

  std::unordered_map<uint32_t, float> score_map;

  for (size_t model_id = 0; model_id < _models.size(); model_id++) {
    auto model_top_k_buckets = top_k_buckets[model_id];
    while (!model_top_k_buckets.empty()) {
      auto bucket_id = model_top_k_buckets.top().second;
      auto bucket_activation = model_top_k_buckets.top().first;

      model_top_k_buckets.pop();
      auto indices = _mach_indices[model_id].getEntities(bucket_id);

      for (const auto& label : indices) {
        if (!score_map.count(label)) {
          score_map[label] = bucket_activation;
        } else {
          score_map[label] += bucket_activation;
        }
      }
    }
  }

  std::vector<std::pair<uint32_t, float>> elements(score_map.begin(),
                                                   score_map.end());
  std::sort(elements.begin(), elements.end(), [](const auto& a, const auto& b) {
    return a.second > b.second;  // sort by value descending
  });

  for (size_t i = 0; i < std::min(k, elements.size()); i++) {
    auto token = _tokenizer->token(elements[i].first);
    if (!_filter_tokens || std::regex_match(token, _allowed_tokens)) {
      if (!decoded.empty()) {
        decoded.push_back(' ');
      }
      decoded.append(token);
    }
  }

  return decoded;
}

ar::ConstArchivePtr MultiSpladeAugmentation::toArchive() const {
  throw std::runtime_error(
      "toArchive is not supported for SpladeAugmentation.");
}

void MultiSpladeConfig::save_stream(std::ostream& output_stream) const {
  (void)_lowercase;
  (void)output_stream;
  throw std::runtime_error(
      "save_stream is not supported for MultiSpladeConfig.");
}

std::shared_ptr<MultiSpladeConfig> MultiSpladeConfig::load_stream(
    std::istream& input_stream) {
  auto archive = ar::deserialize(input_stream);
  (void)archive;

  throw std::runtime_error(
      "load_stream is not supported for MultiSpladeConfig.");
}

}  // namespace thirdai::data