#include "TextClassificationFeaturizer.h"
#include <cereal/archives/binary.hpp>
#include <dataset/src/utils/SafeFileIO.h>
#include <ostream>
#include <string>

namespace thirdai::dataset {

std::vector<std::vector<BoltVector>>
thirdai::dataset::TextClassificationFeaturizer::featurize(
    const std::vector<std::string>& rows) {
  std::vector<std::vector<BoltVector>> feature_columns(getNumDatasets());
  uint32_t label_idx = getNumDatasets() - 1;

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

    switch (_tokens) {
      case Tokens::UNI_ONLY:
        feature_columns[0][row_id] = unigramVector(text_tokens);
      case Tokens::PAIR_ONLY:
        feature_columns[0][row_id] = pairgramVector(text_tokens);
      case Tokens::UNI_PAIR:
        feature_columns[0][row_id] = unigramVector(text_tokens);
        feature_columns[1][row_id] = pairgramVector(text_tokens);
    }

    SegmentedSparseFeatureVector builder;
    _label_block->addVectorSegment(sample, builder);
    feature_columns[label_idx][row_id] = builder.toBoltVector();
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
BoltVector TextClassificationFeaturizer::unigramVector(
    const std::vector<uint32_t>& tokens) {
  BoltVector vector(/* l= */ tokens.size(), /* is_dense= */ false,
                    /* has_gradient= */ false);
  std::copy(tokens.begin(), tokens.end(), vector.active_neurons);
  std::fill_n(vector.activations, vector.len, 1.0);
  return vector;
}
BoltVector TextClassificationFeaturizer::pairgramVector(
    const std::vector<uint32_t>& tokens) {
  std::vector<uint32_t> irc_tokens =
      token_encoding::pairgrams(tokens.data(), tokens.size());

  BoltVector vector(/* l= */ irc_tokens.size(), /* is_dense= */ false,
                    /* has_gradient= */ false);
  std::copy(irc_tokens.begin(), irc_tokens.end(), vector.active_neurons);
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