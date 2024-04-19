#include "SpladeAugmentation.h"
#include <bolt/src/utils/ProgressBar.h>
#include <bolt/src/utils/Timer.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/TextTokenizer.h>
#include <dataset/src/blocks/text/WordpieceTokenizer.h>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>

namespace thirdai::data {

SpladeAugmentation::SpladeAugmentation(std::string input_column,
                                       std::string output_column,
                                       bolt::ModelPtr model,
                                       dataset::WordpieceTokenizerPtr tokenizer,
                                       std::optional<size_t> n_augmented_tokens,
                                       std::optional<float> augmentation_frac,
                                       bool filter_tokens, size_t batch_size,
                                       std::optional<size_t> token_offset)
    : _input_column(std::move(input_column)),
      _output_column(std::move(output_column)),
      _model(std::move(model)),
      _tokenizer(std::move(tokenizer)),
      _n_augmented_tokens(n_augmented_tokens),
      _augmentation_frac(augmentation_frac),
      _filter_tokens(filter_tokens),
      _batch_size(batch_size),
      _token_offset(token_offset) {
  if (_model->inputs().size() != 1 || _model->outputs().size() != 1) {
    throw std::invalid_argument(
        "SpladeAugmentation model must have 1 input and output.");
  }

  if (_model->inputs()[0]->dim() != _tokenizer->vocabSize() ||
      _model->outputs()[0]->dim() != _tokenizer->vocabSize()) {
    throw std::invalid_argument(
        "SpladeAugmentation model input and output dim should match tokenizer "
        "vocab size.");
  }

  if (_n_augmented_tokens.has_value() == _augmentation_frac.has_value()) {
    throw std::invalid_argument(
        "Must specified exactly one of n_augmented_tokens and "
        "augmentation_frac.");
  }
}

ColumnMap SpladeAugmentation::apply(ColumnMap columns, State& state) const {
  (void)state;

  // Note: NGramEncoder with n=1 in this case is a no-op, it is just here to
  // satisfy the encoder requirement.
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
    auto output = _model->forward(batch).at(0);

#pragma omp parallel for default(none) \
    shared(tokenized_text, augmented_text, output, row_index)
    for (size_t i = 0; i < output->batchSize(); i++) {
      augmented_text[row_index + i] = decodeTopTokens(
          output->getVector(i),
          tokensToAdd(tokenized_text->row(row_index + i).size()));
    }

    row_index += output->batchSize();

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

std::string SpladeAugmentation::decodeTopTokens(const BoltVector& vec,
                                                size_t k) const {
  std::string decoded;
  auto topk = vec.topKNeurons(k);
  while (!topk.empty()) { 
    if(!_token_offset){
      auto token = _tokenizer->token(topk.top().second);

      if (!_filter_tokens || std::regex_match(token, _allowed_tokens)) {
        if (!decoded.empty()) {
          decoded.push_back(' ');
        }
        decoded.append(token);
      }
    }else{
      auto token = topk.top().second;

      if (!decoded.empty()) {
          decoded.push_back(' ');
        }
        decoded.append(std::to_string(token + *_token_offset));
    }

    topk.pop();

  }
  return decoded;
}

ar::ConstArchivePtr SpladeAugmentation::toArchive() const {
  auto map = ar::Map::make();

  map->set("input_column", ar::str(_input_column));
  map->set("output_column", ar::str(_output_column));

  map->set("model", _model->toArchive(/*with_optimizer=*/false));
  map->set("tokenize", _tokenizer->toArchive());

  if (_n_augmented_tokens) {
    map->set("n_augmented_tokens", ar::u64(*_n_augmented_tokens));
  }
  if (_augmentation_frac) {
    map->set("augmentation_frac", ar::f32(*_augmentation_frac));
  }
  map->set("filter_tokens", ar::boolean(_filter_tokens));
  map->set("batch_size", ar::u64(_batch_size));
  if(_token_offset){
    map->set("token_offset", ar::u64(*_token_offset));
  }

  return map;
}

SpladeAugmentation::SpladeAugmentation(const ar::Archive& archive)
    : _input_column(archive.str("input_column")),
      _output_column(archive.str("output_column")),
      _model(bolt::Model::fromArchive(*archive.get("model"))),
      _tokenizer(std::make_shared<dataset::WordpieceTokenizer>(
          *archive.get("tokenizer"))),
      _n_augmented_tokens(archive.getOpt<ar::U64>("n_augmented_tokens")),
      _augmentation_frac(archive.getOpt<ar::F32>("augmentation_frac")),
      _filter_tokens(archive.boolean("filter_tokens")),
      _batch_size(archive.u64("batch_size")),
      _token_offset(archive.getOpt<ar::U64>("token_offset")) {}
}  // namespace thirdai::data