#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <optional>
#include <string>
#include <vector>

namespace thirdai::data {

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
                  bool deduplicate,
                  std::optional<uint32_t> output_range = std::nullopt);

  void apply(ColumnMap& column_map, bool /*prepare_for_backpropagate*/) final;

  void backpropagate(ColumnMap& /*columns*/,
                     ContributionColumnMap& /*contribuition_columns*/) final {}

 private:
  columns::SparseArrayColumnPtr deduplicatedUnigramColumn(
      const columns::StringColumnPtr& input_column, uint32_t num_rows);

  columns::TokenArrayColumnPtr rawUnigramColumn(
      const columns::StringColumnPtr& input_column, uint32_t num_rows);

  std::vector<uint32_t> computeUnigrams(const std::string& text);

  // Private constructor for cereal.
  SentenceUnigram();

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  std::string _input_column_name;
  std::string _output_column_name;
  bool _deduplicate;
  std::optional<uint32_t> _output_range;
};

}  // namespace thirdai::data
