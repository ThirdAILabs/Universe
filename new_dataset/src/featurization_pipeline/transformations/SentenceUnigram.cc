#include "SentenceUnigram.h"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <optional>
#include <unordered_map>

namespace thirdai::data {

SentenceUnigram::SentenceUnigram(
    std::string input_column_name, std::string output_column_name,
    bool deduplicate, std::optional<uint32_t> output_range /* = std::nullopt*/)
    : _input_column_name(std::move(input_column_name)),
      _output_column_name(std::move(output_column_name)),
      _deduplicate(deduplicate),
      _output_range(output_range) {}

void SentenceUnigram::apply(ColumnMap& column_map,
                            bool prepare_for_backpropagate) {
  auto input_column = column_map.getStringColumn(_input_column_name);
  uint32_t num_rows = column_map.numRows();

  columns::ColumnPtr output_column;
  if (_deduplicate) {
    output_column = deduplicatedUnigramColumn(input_column, num_rows,
                                              prepare_for_backpropagate);
  } else {
    output_column =
        rawUnigramColumn(input_column, num_rows, prepare_for_backpropagate);
  }
  column_map.setColumn(_output_column_name, output_column);
}

void SentenceUnigram::backpropagate(
    ColumnMap& columns, ContributionColumnMap& contribuition_columns) {
  auto input_column = columns.getStringColumn(_input_column_name);
  auto contribuition_column =
      contribuition_columns.getTokenContributionColumn(_output_column_name);
  uint32_t num_rows = columns.numRows();

  if (!_hash_values) {
    throw std::invalid_argument(
        "in apply method didn't prepare for backpropagation.");
  }
  std::vector<std::vector<columns::Contribution<std::string>>> contributions(
      num_rows);

#pragma omp parallel for default(none) \
    shared(num_rows, contribuition_column, contributions)
  for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
    std::vector<columns::Contribution<uint32_t>> row_token_contributions =
        contribuition_column->getRow(row_idx);

    std::vector<columns::Contribution<std::string>> row_string_contributions(
        row_token_contributions.size());

    for (uint32_t j = 0; j < row_token_contributions.size(); j++) {
      row_string_contributions[j] = columns::Contribution<std::string>(
          _hash_values->at(row_idx)[row_token_contributions[j].value],
          row_token_contributions[j].gradient);
    }
    contributions[row_idx] = row_string_contributions;
  }
  auto input_contribution_column =
      std::make_shared<columns::CppStringContributionColumn>(
          std::move(contributions));

  contribuition_columns.setColumn(_input_column_name,
                                  input_contribution_column);
}

columns::SparseArrayColumnPtr SentenceUnigram::deduplicatedUnigramColumn(
    const columns::StringColumnPtr& input_column, uint32_t num_rows,
    bool prepare_for_backpropagate) {
  std::vector<std::vector<std::pair<uint32_t, float>>> column_values(num_rows);
  std::vector<std::unordered_map<uint32_t, std::string>> hash_values_map(
      num_rows);
  if (prepare_for_backpropagate) {
    hash_values_map.reserve(num_rows);
  }
#pragma omp parallel for default(none)                                       \
    shared(num_rows, column_values, input_column, prepare_for_backpropagate, \
           hash_values_map)
  for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
    std::string text = (*input_column)[row_idx];
    std::vector<uint32_t> unigrams = computeUnigrams(text);
    if (prepare_for_backpropagate) {
      hash_values_map[row_idx] = computeUnigramsMap(text);
    }

    std::vector<std::pair<uint32_t, float>> deduplicated_unigrams;
    // TODO(any): make TextEncodingUtils more usable
    dataset::TextEncodingUtils::sumRepeatedIndices(
        unigrams, /* base_value= */ 1.0, [&](uint32_t unigram, float value) {
          deduplicated_unigrams.push_back(std::make_pair(unigram, value));
        });
    column_values[row_idx] = deduplicated_unigrams;
  }

  if (prepare_for_backpropagate) {
    _hash_values = hash_values_map;
  }

  return std::make_shared<columns::CppSparseArrayColumn>(
      std::move(column_values), _output_range);
}

columns::TokenArrayColumnPtr SentenceUnigram::rawUnigramColumn(
    const columns::StringColumnPtr& input_column, uint32_t num_rows,
    bool prepare_for_backpropagate) {
  std::vector<std::vector<uint32_t>> column_values(num_rows);
  std::vector<std::unordered_map<uint32_t, std::string>> hash_values_map(
      num_rows);
  if (prepare_for_backpropagate) {
    hash_values_map.reserve(num_rows);
  }
#pragma omp parallel for default(none)                                       \
    shared(num_rows, column_values, input_column, prepare_for_backpropagate, \
           hash_values_map)
  for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
    std::string text = (*input_column)[row_idx];
    std::vector<uint32_t> unigrams = computeUnigrams(text);
    if (prepare_for_backpropagate) {
      hash_values_map[row_idx] = computeUnigramsMap(text);
    }
    column_values[row_idx] = unigrams;
  }
  if (prepare_for_backpropagate) {
    _hash_values = hash_values_map;
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

std::unordered_map<uint32_t, std::string> SentenceUnigram::computeUnigramsMap(
    const std::string& text) {
  std::unordered_map<uint32_t, std::string> unigrams_map;
  if (_output_range) {
    unigrams_map = dataset::TextEncodingUtils::buildUnigramHashToWordMap(
        text, *_output_range);
  } else {
    unigrams_map =
        dataset::TextEncodingUtils::buildRawUnigramHashToWordMap(text);
  }
  return unigrams_map;
}

// Private constructor for cereal.
SentenceUnigram::SentenceUnigram()
    : _input_column_name(),
      _output_column_name(),
      _deduplicate(false),
      _output_range(std::nullopt),
      _hash_values(std::nullopt) {}

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
