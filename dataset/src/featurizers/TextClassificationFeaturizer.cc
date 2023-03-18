#include "TextClassificationFeaturizer.h"
#include <cereal/archives/binary.hpp>
#include <dataset/src/utils/SafeFileIO.h>
#include <algorithm>
#include <ostream>
#include <string>

namespace thirdai::dataset {

std::vector<std::vector<BoltVector>>
thirdai::dataset::TextClassificationFeaturizer::featurize(
    const std::vector<std::string>& rows) {
  std::vector<std::vector<BoltVector>> feature_columns(N_DATASETS);

  for (auto& column : feature_columns) {
    column.resize(rows.size());
  }

#pragma omp parallel for default(none) shared( \
    rows, feature_columns, label_idx, _delimiter, _tokens, _label_block)
  for (uint32_t row_id = 0; row_id < rows.size(); row_id++) {
    const std::string& row = rows[row_id];

    CsvSampleRef sample(row, _delimiter);
    auto text_column = sample.column(_text_column);

    std::vector<uint32_t> text_tokens = tokens(text_column);

    feature_columns[LRC_DATASET_ID][row_id] = lrcVector(text_tokens);
    feature_columns[IRC_DATASET_ID][row_id] = ircVector(text_tokens);
    feature_columns[SRC_DATASET_ID][row_id] = srcVector(text_tokens);

    SegmentedSparseFeatureVector builder;
    _label_block->addVectorSegment(sample, builder);
    feature_columns[LABEL_DATASET_ID][row_id] = builder.toBoltVector();
  }

  return feature_columns;
}

std::vector<uint32_t> TextClassificationFeaturizer::tokens(
    std::string_view text_column) {
  auto token_strings = text::split(text_column, /* delimiter= */ ' ');
  std::vector<uint32_t> token_ints;
  token_ints.reserve(token_strings.size());
  for (const auto& str : token_strings) {
    token_ints.push_back(std::strtoul(str.data(), nullptr, 10));
  }
  return token_ints;
}

BoltVector TextClassificationFeaturizer::lrcVector(
    const std::vector<uint32_t>& tokens) const {
  size_t n_lrc_tokens = std::min(tokens.size(), _lrc_len);
  BoltVector vector(/* l= */ n_lrc_tokens, /* is_dense= */ false,
                    /* has_gradient= */ false);
  std::copy(tokens.end() - n_lrc_tokens, tokens.end(), vector.active_neurons);
  std::fill_n(vector.activations, vector.len, 1.0);
  return vector;
}

BoltVector TextClassificationFeaturizer::ircVector(
    const std::vector<uint32_t>& tokens) const {
  size_t n_irc_tokens = std::min(tokens.size(), _irc_len);
  size_t start_idx = tokens.size() - n_irc_tokens;
  std::vector<uint32_t> irc_tokens =
      token_encoding::pairgrams(tokens.data() + start_idx, n_irc_tokens);

  BoltVector vector(/* l= */ irc_tokens.size(), /* is_dense= */ false,
                    /* has_gradient= */ false);
  std::copy(irc_tokens.begin(), irc_tokens.end(), vector.active_neurons);
  std::fill_n(vector.activations, vector.len, 1.0);
  return vector;
}

BoltVector TextClassificationFeaturizer::srcVector(
    const std::vector<uint32_t>& tokens) const {
  BoltVector vector(/* l= */ _src_len, /* is_dense= */ false,
                    /* has_gradient= */ false);

  size_t n_src_tokens = std::min(tokens.size(), _src_len);
  uint32_t n_padding = _src_len - n_src_tokens;

  std::fill_n(vector.active_neurons, n_padding, 0);
  std::copy(tokens.end() - n_src_tokens, tokens.end(),
            vector.active_neurons + n_padding);
  std::fill_n(vector.activations, vector.len, 1.0);
  return vector;
}

void TextClassificationFeaturizer::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void TextClassificationFeaturizer::save_stream(
    std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

TextClassificationFeaturizerPtr TextClassificationFeaturizer::load(
    const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

TextClassificationFeaturizerPtr TextClassificationFeaturizer::load_stream(
    std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<TextClassificationFeaturizer> deserialize_into(
      new TextClassificationFeaturizer());
  iarchive(*deserialize_into);
  return deserialize_into;
}

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TextClassificationFeaturizer)