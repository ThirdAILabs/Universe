#include "CrossColumnPairgrams.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/MurmurHash.h>
#include <data/src/columns/ArrayColumns.h>
#include <dataset/src/utils/TokenEncoding.h>

namespace thirdai::data {

CrossColumnPairgrams::CrossColumnPairgrams(
    std::vector<std::string> input_column_names, std::string output_column_name,
    size_t hash_range)
    : _input_column_names(std::move(input_column_names)),
      _output_column_name(std::move(output_column_name)),
      _hash_range(hash_range) {}

ColumnMap CrossColumnPairgrams::apply(ColumnMap columns, State& state) const {
  (void)state;

  std::vector<std::pair<ValueColumnBasePtr<uint32_t>, uint32_t>> token_columns;

  for (const auto& col_name : _input_column_names) {
    uint32_t column_seed = dataset::token_encoding::seededMurmurHash(
        /* key = */ col_name.c_str(), /* len = */ col_name.size());

    token_columns.emplace_back(columns.getValueColumn<uint32_t>(col_name),
                               column_seed);
  }

  uint32_t num_rows = columns.numRows();
  std::vector<std::vector<uint32_t>> cross_column_pairgrams(num_rows);
#pragma omp parallel for default(none) \
    shared(num_rows, token_columns, cross_column_pairgrams) if (num_rows > 1)
  for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
    std::vector<uint32_t> salted_unigrams;
    salted_unigrams.reserve(token_columns.size());

    for (const auto& [column, column_seed] : token_columns) {
      salted_unigrams.push_back(hashToken(column->value(row_idx), column_seed));
    }

    // We don't deduplicate pairgrams since we ensure unique hash values above,
    // thus reducing the chance of duplicates.
    auto pairgrams = dataset::token_encoding::pairgrams(salted_unigrams);
    dataset::token_encoding::mod(pairgrams, _hash_range);

    cross_column_pairgrams[row_idx] = std::move(pairgrams);
  }

  auto output_column = ArrayColumn<uint32_t>::make(
      std::move(cross_column_pairgrams), _hash_range);

  columns.setColumn(_output_column_name, output_column);

  return columns;
}

uint32_t CrossColumnPairgrams::hashToken(uint32_t token, uint32_t column_seed) {
  return hashing::MurmurHash(reinterpret_cast<const char*>(&token),
                             sizeof(uint32_t), column_seed);
}

void CrossColumnPairgrams::buildExplanationMap(
    const ColumnMap& input, State& state, ExplanationMap& explanations) const {
  std::vector<uint32_t> tokens;
  for (const auto& column : _input_column_names) {
    tokens.push_back(input.getValueColumn<uint32_t>(column)->value(0));
  }

  auto output = apply(input, state);

  auto output_tokens =
      output.getArrayColumn<uint32_t>(_output_column_name)->row(0);

  size_t output_idx = 0;
  for (size_t col_a = 0; col_a < tokens.size(); col_a++) {
    for (size_t col_b = 0; col_b <= col_a; col_b++) {
      uint32_t token = tokens[col_a];
      uint32_t prev_token = tokens[col_b];

      auto token_exp =
          explanations.explain(_input_column_names.at(col_a), token);

      auto prev_token_exp =
          explanations.explain(_input_column_names.at(col_b), prev_token);

      auto output_token = output_tokens[output_idx++];

      if (col_a == col_b) {
        explanations.store(_output_column_name, output_token, token_exp);
      } else {
        auto explanation = prev_token_exp.append(" and ").append(token_exp);
        explanations.store(_output_column_name, output_token, explanation);
      }
    }
  }
}

template void CrossColumnPairgrams::serialize(cereal::BinaryInputArchive&);
template void CrossColumnPairgrams::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void CrossColumnPairgrams::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column_names,
          _output_column_name, _hash_range);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::CrossColumnPairgrams)
