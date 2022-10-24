#pragma once

#include <hashing/src/MurmurHash.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <optional>
#include <string>
#include <vector>

namespace thirdai::dataset {

/**
 * @brief This transformation assumes as input a StringValueColumn with
 * sentences. It hashes each word (space delimited) and records the result in
 * either a VectorSparseValueColumn or VectorIndexValueArrayColumn depending on
 * whether the deduplicate flag is true or false.
 *
 * @param output_range If specified, each word hash will be modded by
 * output_range. Defaults to std::nullopt.
 * @param deduplicate If true, deduplicates any repeated uint32_t hashes by
 * storing a vector of (index, value) pairs where value is the number of times
 * each original hash index appears. Otherwise returns the original vector of
 * uint32_t unigram tokens into the column
 */
class SentenceUnigram : public Transformation {
 public:
  SentenceUnigram(std::string input_column_name, std::string output_column_name,
                  std::optional<uint32_t> output_range = std::nullopt,
                  bool deduplicate = false)
      : _input_column_name(std::move(input_column_name)),
        _output_column_name(std::move(output_column_name)),
        _output_range(output_range),
        _deduplicate(deduplicate) {}

  void apply(ColumnMap& column_map) final {
    auto input_column = column_map.getStringColumn(_input_column_name);
    uint32_t num_rows = column_map.numRows();

    ColumnPtr output_column;
    if (_deduplicate) {
      output_column = deduplicatedUnigramColumn(input_column, num_rows);
    } else {
      output_column = unigramTokenColumn(input_column, num_rows);
    }
    column_map.setColumn(_output_column_name, output_column);
  }

 private:
  IndexValueArrayColumnPtr deduplicatedUnigramColumn(
      const StringColumnPtr& input_column, uint32_t num_rows) {
    std::vector<std::vector<std::pair<uint32_t, float>>> column_values(
        num_rows);
#pragma omp parallel for default(none) \
    shared(num_rows, column_values, input_column, _output_range)
    for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
      std::string text = (*input_column)[row_idx];
      std::vector<uint32_t> unigrams = computeUnigrams(text);

      std::vector<std::pair<uint32_t, float>> deduplicated_unigrams(
          unigrams.size());
      TextEncodingUtils::sumRepeatedIndices(
          unigrams, /* base_value= */ 1.0, [&](uint32_t unigram, float value) {
            deduplicated_unigrams.push_back(std::make_pair(unigram, value));
          });
      column_values[row_idx] = deduplicated_unigrams;
    }

    return std::make_shared<VectorIndexValueArrayColumn>(
        std::move(column_values));
  }

  SparseArrayColumnPtr unigramTokenColumn(const StringColumnPtr& input_column,
                                          uint32_t num_rows) {
    std::vector<std::vector<uint32_t>> column_values(num_rows);
#pragma omp parallel for default(none) \
    shared(num_rows, column_values, input_column, _output_range)
    for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
      std::string text = (*input_column)[row_idx];
      std::vector<uint32_t> unigrams = computeUnigrams(text);
      column_values[row_idx] = unigrams;
    }

    return std::make_shared<VectorSparseArrayColumn>(std::move(column_values));
  }

  std::vector<uint32_t> computeUnigrams(const std::string& text) {
    std::vector<uint32_t> unigrams;
    if (_output_range) {
      unigrams =
          TextEncodingUtils::computeRawUnigramsWithRange(text, *_output_range);
    } else {
      unigrams = TextEncodingUtils::computeRawUnigrams(text);
    }
    return unigrams;
  }

  std::string _input_column_name;
  std::string _output_column_name;
  std::optional<uint32_t> _output_range;
  bool _deduplicate;
};

}  // namespace thirdai::dataset