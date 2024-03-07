#include "SpladeAugmentation.h"
#include <bolt/src/utils/ProgressBar.h>
#include <bolt/src/utils/Timer.h>
#include <bolt_vector/src/BoltVector.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/TextTokenizer.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::data {

SpladeAugmentation::SpladeAugmentation(std::string input_column,
                                       std::string output_column,
                                       const std::string& model_checkpoint,
                                       const std::string& tokenizer_vocab,
                                       size_t n_augmented_tokens,
                                       bool lowercase)
    : SpladeAugmentation(/*input_column=*/std::move(input_column),
                         /*output_column=*/std::move(output_column),
                         /*model=*/bolt::Model::load(model_checkpoint),
                         /*tokenizer=*/
                         std::make_shared<dataset::WordpieceTokenizer>(
                             tokenizer_vocab, lowercase),
                         /*n_augmented_tokens=*/n_augmented_tokens) {}

SpladeAugmentation::SpladeAugmentation(std::string input_column,
                                       std::string output_column,
                                       bolt::ModelPtr model,
                                       dataset::WordpieceTokenizerPtr tokenizer,
                                       size_t n_augmented_tokens)
    : _input_column(std::move(input_column)),
      _output_column(std::move(output_column)),
      _model(std::move(model)),
      _tokenizer(std::move(tokenizer)),
      _n_augmented_tokens(n_augmented_tokens) {
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
}

ColumnMap SpladeAugmentation::apply(ColumnMap columns, State& state) const {
  (void)state;

  data::TextTokenizer tokenizer(
      /*input_column=*/_input_column, /*output_indices=*/_input_column,
      /*output_values=*/std::nullopt, /*tokenizer=*/_tokenizer,
      /*encoder=*/dataset::NGramEncoder::make(1), /*lowercase=*/false,
      /*dim=*/_tokenizer->vocabSize());

  auto tokenized_columns = tokenizer.applyStateless(columns);

  auto batches = data::toTensorBatches(tokenized_columns,
                                       {OutputColumns(_input_column)}, 4096);

  auto input_text = columns.getValueColumn<std::string>(_input_column);

  std::vector<std::string> augmented_text(input_text->numRows());

  ProgressBar bar("augmenting data", batches.size());
  bolt::utils::Timer timer;
  size_t row_index = 0;
  for (const auto& batch : batches) {
    auto output = _model->forward(batch).at(0);

#pragma omp parallel for default(none) \
    shared(input_text, augmented_text, output, row_index)
    for (size_t i = 0; i < output->batchSize(); i++) {
      augmented_text[row_index + i] =
          input_text->value(row_index + i) +
          decodeTopTokens(output->getVector(i), _n_augmented_tokens);
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
  auto topk = vec.findKLargestActivations(k);
  while (!topk.empty()) {
    decoded.push_back(' ');
    decoded.append(_tokenizer->token(topk.top().second));
    topk.pop();
  }
  return decoded;
}

ar::ConstArchivePtr SpladeAugmentation::toArchive() const {
  throw std::runtime_error(
      "toArchive is not supported for SpladeAugmentation.");
}

}  // namespace thirdai::data