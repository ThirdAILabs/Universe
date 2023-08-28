#include "TextTokenizer.h"
#include <data/src/columns/ArrayColumns.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/blocks/text/WordpieceTokenizer.h>
#include <proto/tokenizers.pb.h>
#include <stdexcept>
#include <string>

namespace thirdai::data {

TextTokenizer::TextTokenizer(std::string input_column,
                             std::string output_column,
                             dataset::TextTokenizerPtr tokenizer,
                             dataset::TextEncoderPtr encoder, bool lowercase,
                             size_t dim)
    : _input_column(std::move(input_column)),
      _output_column(std::move(output_column)),
      _tokenizer(std::move(tokenizer)),
      _encoder(std::move(encoder)),
      _lowercase(lowercase),
      _dim(dim) {}

TextTokenizer::TextTokenizer(const proto::data::TextTokenizer& text)
    : _input_column(text.input_column()),
      _output_column(text.output_column()),
      _lowercase(text.lowercase()),
      _dim(text.dim()) {
  switch (text.tokenizer().tokenizer_case()) {
    case proto::data::Tokenizer::kWordpiece:
      _tokenizer = std::make_shared<dataset::WordpieceTokenizer>(
          text.tokenizer().wordpiece());
      break;
    case proto::data::Tokenizer::kSplit:
      _tokenizer = dataset::NaiveSplitTokenizer::make(
          text.tokenizer().split().delimiter());
      break;
    case proto::data::Tokenizer::kWordPunct:
      _tokenizer = dataset::WordPunctTokenizer::make();
      break;
    case proto::data::Tokenizer::kCharKgram:
      _tokenizer =
          dataset::CharKGramTokenizer::make(text.tokenizer().char_kgram().k());
      break;
    default:
      throw std::invalid_argument(
          "Invalid text tokenizer specified in fromProto.");
  }

  switch (text.encoder().encoder_case()) {
    case proto::data::TextEncoder::kNgram:
      _encoder = dataset::NGramEncoder::make(text.encoder().ngram().n());
      break;
    case proto::data::TextEncoder::kPairgram:
      _encoder = dataset::PairGramEncoder::make();
      break;
    default:
      throw std::invalid_argument(
          "Invalid text encoder specified in fromProto.");
  }
}

ColumnMap TextTokenizer::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto text_col = columns.getValueColumn<std::string>(_input_column);

  std::vector<std::vector<uint32_t>> output_tokens(text_col->numRows());

#pragma omp parallel for default(none) \
    shared(text_col, output_tokens) if (columns.numRows() > 1)
  for (size_t i = 0; i < text_col->numRows(); i++) {
    std::string string = text_col->value(i);

    if (_lowercase) {
      string = text::lower(string);
    }

    std::vector<uint32_t> tokens = _tokenizer->tokenize(string);
    std::vector<uint32_t> indices = _encoder->encode(tokens);
    dataset::token_encoding::mod(indices, _dim);

    output_tokens[i] = std::move(indices);
  }

  auto token_col = ArrayColumn<uint32_t>::make(std::move(output_tokens), _dim);

  columns.setColumn(_output_column, token_col);
  return columns;
}

void TextTokenizer::buildExplanationMap(const ColumnMap& input, State& state,
                                        ExplanationMap& explanations) const {
  (void)state;

  const std::string& text =
      input.getValueColumn<std::string>(_input_column)->value(0);

  std::vector<uint32_t> tokens = _tokenizer->tokenize(text);
  std::vector<uint32_t> indices = _encoder->encode(tokens);
  dataset::token_encoding::mod(indices, _dim);

  for (const auto& index : indices) {
    uint32_t token = _encoder->undoEncoding(tokens, index, _dim);
    auto word = _tokenizer->getResponsibleWord(text, token);

    explanations.store(_output_column, index,
                       "word '" + word + "' from " +
                           explanations.explain(_input_column, text));
  }
}

proto::data::Transformation* TextTokenizer::toProto() const {
  auto* transformation = new proto::data::Transformation();
  auto* text = transformation->mutable_text_tokenizer();

  text->set_input_column(_input_column);
  text->set_output_column(_output_column);
  text->set_allocated_tokenizer(_tokenizer->toProto());
  text->set_allocated_encoder(_encoder->toProto());
  text->set_lowercase(_lowercase);
  text->set_dim(_dim);

  return transformation;
}

}  // namespace thirdai::data
