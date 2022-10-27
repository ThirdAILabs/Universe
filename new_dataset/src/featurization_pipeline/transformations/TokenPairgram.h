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
#include <vector>

namespace thirdai::dataset {

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

  void apply(ColumnMap& column_map) final {
    SparseArrayColumnPtr input_column =
        column_map.getSparseArrayColumn(_input_column_name);
    uint32_t num_rows = column_map.numRows();

    std::vector<std::vector<std::pair<uint32_t, float>>> column_values(
        num_rows);

    // set here because pragma doesnt like sharing private member variables
    uint32_t output_range = _output_range;
#pragma omp parallel for default(none) \
    shared(num_rows, column_values, input_column, output_range)
    for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
      ArrayColumn<uint32_t>::RowReference input_tokens_buffer =
          (*input_column)[row_idx];
      std::vector<uint32_t> input_tokens_vector(input_tokens_buffer.begin(),
                                                input_tokens_buffer.end());
      std::vector<uint32_t> pairgrams =
          TextEncodingUtils::computeRawPairgramsFromUnigrams(
              input_tokens_vector, _output_range);

      std::vector<std::pair<uint32_t, float>> deduplicated_pairgrams;
      TextEncodingUtils::sumRepeatedIndices(
          pairgrams, /* base_value= */ 1.0,
          [&](uint32_t pairgram, float value) {
            deduplicated_pairgrams.push_back(std::make_pair(pairgram, value));
          });
      column_values[row_idx] = deduplicated_pairgrams;
    }

    auto output_column = std::make_shared<VectorIndexValueArrayColumn>(
        std::move(column_values), _output_range);
    column_map.setColumn(_output_column_name, output_column);
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
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TokenPairgram)
