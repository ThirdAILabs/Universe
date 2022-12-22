#include "SentenceUnigram.h"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>

namespace thirdai::data {

SentenceUnigram::SentenceUnigram(
    std::string input_column_name, std::string output_column_name,
    bool deduplicate, std::optional<uint32_t> output_range /* = std::nullopt*/)
    : _input_column_name(std::move(input_column_name)),
      _output_column_name(std::move(output_column_name)),
      _deduplicate(deduplicate),
      _output_range(output_range) {}

void SentenceUnigram::apply(ColumnMap& column_map,bool  /*prepare_for_backpropagate*/) {
  auto input_column = column_map.getStringColumn(_input_column_name);
  uint32_t num_rows = column_map.numRows();

  columns::ColumnPtr output_column;
  if (_deduplicate) {
    output_column = deduplicatedUnigramColumn(input_column, num_rows);
  } else {
    output_column = rawUnigramColumn(input_column, num_rows);
  }
  column_map.setColumn(_output_column_name, output_column);
}

columns::SparseArrayColumnPtr SentenceUnigram::deduplicatedUnigramColumn(
    const columns::StringColumnPtr& input_column, uint32_t num_rows) {
  std::vector<std::vector<std::pair<uint32_t, float>>> column_values(num_rows);
#pragma omp parallel for default(none) \
    shared(num_rows, column_values, input_column)
  for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
    std::string text = (*input_column)[row_idx];
    std::vector<uint32_t> unigrams = computeUnigrams(text);

    std::vector<std::pair<uint32_t, float>> deduplicated_unigrams;
    // TODO(any): make TextEncodingUtils more usable
    dataset::TextEncodingUtils::sumRepeatedIndices(
        unigrams, /* base_value= */ 1.0, [&](uint32_t unigram, float value) {
          deduplicated_unigrams.push_back(std::make_pair(unigram, value));
        });
    column_values[row_idx] = deduplicated_unigrams;
  }

  return std::make_shared<columns::CppSparseArrayColumn>(
      std::move(column_values), _output_range);
}

columns::TokenArrayColumnPtr SentenceUnigram::rawUnigramColumn(
    const columns::StringColumnPtr& input_column, uint32_t num_rows) {
  std::vector<std::vector<uint32_t>> column_values(num_rows);
#pragma omp parallel for default(none) \
    shared(num_rows, column_values, input_column)
  for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
    std::string text = (*input_column)[row_idx];
    std::vector<uint32_t> unigrams = computeUnigrams(text);
    column_values[row_idx] = unigrams;
  }

  return std::make_shared<columns::CppTokenArrayColumn>(
      std::move(column_values), _output_range);
}

std::vector<uint32_t> SentenceUnigram::computeUnigrams(
    const std::string& text) {
  std::vector<uint32_t> unigrams;
  if (_output_range) {
    unigrams = dataset::TextEncodingUtils::computeRawUnigramsWithRange(
        text, *_output_range);
  } else {
    unigrams = dataset::TextEncodingUtils::computeRawUnigrams(text);
  }
  return unigrams;
}

// Private constructor for cereal.
SentenceUnigram::SentenceUnigram()
    : _input_column_name(),
      _output_column_name(),
      _deduplicate(false),
      _output_range(std::nullopt) {}

template <class Archive>
void SentenceUnigram::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column_name,
          _output_column_name, _deduplicate, _output_range);
}

template void SentenceUnigram::serialize<cereal::BinaryInputArchive>(
    cereal::BinaryInputArchive&);

template void SentenceUnigram::serialize<cereal::BinaryOutputArchive>(
    cereal::BinaryOutputArchive&);

template void SentenceUnigram::serialize<cereal::PortableBinaryInputArchive>(
    cereal::PortableBinaryInputArchive&);

template void SentenceUnigram::serialize<cereal::PortableBinaryOutputArchive>(
    cereal::PortableBinaryOutputArchive&);

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::SentenceUnigram)
