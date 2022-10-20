#pragma once

#include <hashing/src/MurmurHash.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <string>
#include <vector>

namespace thirdai::dataset {

class TokenPairgram : public Transformation {
  TokenPairgram(std::string input_column_name, std::string output_column_name,
                uint32_t output_range)
      : _input_column_name(std::move(input_column_name)),
        _output_column_name(std::move(output_column_name)),
        _output_range(output_range) {}

  void apply(ColumnMap& column_map) final {
    auto column = column_map.getSparseArrayColumn(_input_column_name);
    uint32_t num_rows = column_map.numRows();

#pragma omp parallel for default(none) shared(num_rows, column, _output_range)
    for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
      auto input_tokens_buffer = (*column)[row_idx];
      std::vector<uint32_t> input_tokens_vector(input_tokens_buffer.begin(),
                                                input_tokens_buffer.end());
      std::vector<uint32_t> pairgrams =
          TextEncodingUtils::computeRawPairgramsFromUnigrams(
              input_tokens_vector, _output_range);

      TextEncodingUtils::sumRepeatedIndices(
          pairgrams, /* base_value= */ 1.0, [&](uint32_t pairgram, float value) {
            vec.addSparseFeatureToSegment(pairgram, value);
          });
    }
  }

  std::string _input_column_name;
  std::string _output_column_name;
  uint32_t _output_range;
};

}  // namespace thirdai::dataset