#include "SpladeAugmentation.h"
#include <bolt/src/utils/ProgressBar.h>
#include <bolt/src/utils/Timer.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/TextTokenizer.h>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>

namespace thirdai::data {

SpladeConfig::SpladeConfig(const std::string& model_checkpoint,
                           const std::string& tokenizer_vocab,
                           std::optional<size_t> n_augmented_tokens,
                           std::optional<float> augmentation_frac,
                           bool filter_tokens, size_t batch_size,
                           bool lowercase,
                           std::optional<uint32_t> strong_sample_override)
    : model(bolt::Model::load(model_checkpoint)),
      tokenizer(std::make_shared<dataset::WordpieceTokenizer>(tokenizer_vocab,
                                                              lowercase)),
      n_augmented_tokens(n_augmented_tokens),
      augmentation_frac(augmentation_frac),
      filter_tokens(filter_tokens),
      batch_size(batch_size),
      strong_sample_override(strong_sample_override),
      _model_checkpoint(model_checkpoint),
      _tokenizer_vocab(tokenizer_vocab),
      _lowercase(lowercase) {}

SpladeAugmentation::SpladeAugmentation(std::string input_column,
                                       std::string output_column,
                                       const SpladeConfig& config)
    : SpladeAugmentation(/*input_column=*/std::move(input_column),
                         /*output_column=*/std::move(output_column),
                         /*model=*/config.model,
                         /*tokenizer=*/config.tokenizer,
                         /*n_augmented_tokens=*/config.n_augmented_tokens,
                         /*augmentation_frac=*/config.augmentation_frac,
                         /*filter_tokens=*/config.filter_tokens,
                         /*batch_size=*/config.batch_size) {}

SpladeAugmentation::SpladeAugmentation(std::string input_column,
                                       std::string output_column,
                                       bolt::ModelPtr model,
                                       dataset::WordpieceTokenizerPtr tokenizer,
                                       std::optional<size_t> n_augmented_tokens,
                                       std::optional<float> augmentation_frac,
                                       bool filter_tokens, size_t batch_size)
    : _input_column(std::move(input_column)),
      _output_column(std::move(output_column)),
      _model(std::move(model)),
      _tokenizer(std::move(tokenizer)),
      _n_augmented_tokens(n_augmented_tokens),
      _augmentation_frac(augmentation_frac),
      _filter_tokens(filter_tokens),
      _batch_size(batch_size) {
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
  output.dropColumn(_input_column);
  output.setColumn(_output_column,
                   ValueColumn<std::string>::make(std::move(augmented_text)));

  return output;
}

std::string SpladeAugmentation::decodeTopTokens(const BoltVector& vec,
                                                size_t k) const {
  std::string decoded;
  auto topk = vec.topKNeurons(k);
  while (!topk.empty()) {
    auto token = _tokenizer->token(topk.top().second);
    topk.pop();

    if (!_filter_tokens || std::regex_match(token, _allowed_tokens)) {
      if (!decoded.empty()) {
        decoded.push_back(' ');
      }
      decoded.append(token);
    }
  }
  return decoded;
}

ar::ConstArchivePtr SpladeAugmentation::toArchive() const {
  throw std::runtime_error(
      "toArchive is not supported for SpladeAugmentation.");
}

void SpladeConfig::save_stream(std::ostream& output_stream) const {
  auto map = ar::Map::make();
  map->set("model_checkpoint", ar::str(_model_checkpoint));
  map->set("tokenizer_vocab", ar::str(_tokenizer_vocab));
  if (n_augmented_tokens) {
    map->set("n_augmented_tokens", ar::u64(*n_augmented_tokens));
  }
  if (augmentation_frac) {
    map->set("augmentation_frac", ar::f32(*augmentation_frac));
  }
  map->set("batch_size", ar::u64(batch_size));
  map->set("lowercase", ar::boolean(_lowercase));

  ar::serialize(map, output_stream);
}

std::shared_ptr<SpladeConfig> SpladeConfig::load_stream(
    std::istream& input_stream) {
  auto archive = ar::deserialize(input_stream);

  return std::make_shared<SpladeConfig>(
      archive->str("model_checkpoint"), archive->str("tokenizer_vocab"),
      archive->getOpt<ar::U64>("n_augmented_tokens"),
      archive->getOpt<ar::F32>("augmentation_frac"), archive->u64("batch_size"),
      archive->boolean("lowercase"));
}

}  // namespace thirdai::data