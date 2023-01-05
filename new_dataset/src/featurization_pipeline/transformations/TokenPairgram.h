#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::data {

/**
 * @brief This transformation assumes as input a SparseArrayColumn, computes
 * pairgrams for each row, deduplicates common indices, and returns the results
 * as a new IndexValueArrayColumn.
 */
class TokenPairgram : public Transformation {
 public:
  TokenPairgram(std::string input_column_name, std::string output_column_name,
                uint32_t output_range)
      : _input_column_name(std::move(input_column_name)),
        _output_column_name(std::move(output_column_name)),
        _output_range(output_range) {}

  void apply(ColumnMap& column_map, bool prepare_for_backpropagate) final {
    columns::TokenArrayColumnPtr input_column =
        column_map.getTokenArrayColumn(_input_column_name);
    uint32_t num_rows = column_map.numRows();

    std::vector<std::vector<std::pair<uint32_t, float>>> column_values(
        num_rows);

    std::vector<std::vector<dataset::TextEncodingUtils::PairGram>>
        pairgrams_info;

    if (prepare_for_backpropagate) {
      pairgrams_info.resize(num_rows);
    }

#pragma omp parallel for default(none)                                       \
    shared(num_rows, column_values, input_column, prepare_for_backpropagate, \
           pairgrams_info)
    for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
      columns::ArrayColumn<uint32_t>::RowReference input_tokens_buffer =
          (*input_column)[row_idx];
      std::vector<uint32_t> input_tokens_vector(input_tokens_buffer.begin(),
                                                input_tokens_buffer.end());
      std::vector<uint32_t> pairgrams =
          dataset::TextEncodingUtils::computeRawPairgramsFromUnigrams(
              input_tokens_vector, _output_range);

      if (prepare_for_backpropagate) {
        pairgrams_info[row_idx] = dataset::TextEncodingUtils::returnPairgrams(
            input_tokens_vector, _output_range);
      }

      std::vector<std::pair<uint32_t, float>> deduplicated_pairgrams;
      dataset::TextEncodingUtils::sumRepeatedIndices(
          pairgrams, /* base_value= */ 1.0,
          [&](uint32_t pairgram, float value) {
            deduplicated_pairgrams.push_back(std::make_pair(pairgram, value));
          });
      column_values[row_idx] = deduplicated_pairgrams;
    }

    if (prepare_for_backpropagate) {
      std::vector<std::unordered_map<uint32_t, std::vector<uint32_t>>>
          pairgrams(num_rows);
#pragma omp parallel for default(none) shared(pairgrams_info, pairgrams)
      for (uint32_t i = 0; i < pairgrams_info.size(); i++) {
        std::unordered_map<uint32_t, std::vector<uint32_t>> pairgram_map;
        for (auto& j : pairgrams_info[i]) {
          pairgram_map[j.pairgram].push_back(j.first_token);
          pairgram_map[j.pairgram].push_back(j.second_token);
        }
        pairgrams[i] = pairgram_map;
      }
      _pairgrams = pairgrams;
    }

    auto output_column = std::make_shared<columns::CppSparseArrayColumn>(
        std::move(column_values), _output_range);
    column_map.setColumn(_output_column_name, output_column);
  }

  void backpropagate(ColumnMap& columns,
                     ContributionColumnMap& contribuition_columns) final {
    columns::TokenArrayColumnPtr input_column =
        columns.getTokenArrayColumn(_input_column_name);
    uint32_t num_rows = columns.numRows();
    if (!contribuition_columns.checkColumnExists(_output_column_name)) {
      return;
    }
    if (!_pairgrams) {
      throw std::invalid_argument(
          "in apply method didn't prepare for backpropagation.");
    }
    auto contribuition_column =
        contribuition_columns.getTokenContributionColumn(_output_column_name);

    std::vector<std::vector<columns::Contribution<uint32_t>>> contributions(
        num_rows);
#pragma omp parallel for default(none) \
    shared(num_rows, contribuition_column, input_column, contributions)
    for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
      std::vector<columns::Contribution<uint32_t>> row_token_contributions =
          contribuition_column->getRow(row_idx);

      columns::ArrayColumn<uint32_t>::RowReference input_tokens_buffer =
          (*input_column)[row_idx];
      std::vector<uint32_t> input_tokens_vector(input_tokens_buffer.begin(),
                                                input_tokens_buffer.end());

      std::vector<columns::Contribution<uint32_t>> source_row_contributions(
          input_tokens_vector.size());

      std::unordered_map<uint32_t, float> individual_contributions;

      for (auto input_token : input_tokens_vector) {
        individual_contributions[input_token] = 0;
      }

      for (auto& row_token_contribution : row_token_contributions) {
        for (auto val : _pairgrams->at(row_idx)[row_token_contribution.value]) {
          individual_contributions[val] +=
              (row_token_contribution.gradient) /
              (_pairgrams->at(row_idx)[row_token_contribution.value].size());
        }
      }

      // (TODO):have to see the percentages absolute sum to be 100 percent.
      for (uint32_t i = 0; i < input_tokens_vector.size(); i++) {
        source_row_contributions[i] = {
            input_tokens_vector[i],
            individual_contributions[input_tokens_vector[i]]};
      }
      contributions[row_idx] = source_row_contributions;
    }
    auto input_contribution_column =
        std::make_shared<columns::CppTokenContributionColumn>(
            std::move(contributions));

    contribuition_columns.setColumn(_input_column_name,
                                    input_contribution_column);
  }

 private:
  // Private constructor for cereal.
  TokenPairgram()
      : _input_column_name(), _output_column_name(), _output_range(0) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Transformation>(this), _input_column_name,
            _output_column_name, _output_range);
  }

  std::string _input_column_name;
  std::string _output_column_name;
  uint32_t _output_range;
  std::optional<
      std::vector<std::unordered_map<uint32_t, std::vector<uint32_t>>>>
      _pairgrams;
};

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::TokenPairgram)
