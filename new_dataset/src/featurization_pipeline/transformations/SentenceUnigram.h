#pragma once

#include <hashing/src/MurmurHash.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <optional>
#include <string>
#include <vector>

namespace thirdai::dataset {

class SentenceUnigram : public Transformation {
  SentenceUnigram(std::string input_column_name, std::string output_column_name,
                  std::optional<uint32_t> output_range)
      : _input_column_name(std::move(input_column_name)),
        _output_column_name(std::move(output_column_name)),
        _output_range(output_range) {}

  void apply(ColumnMap& column_map) final {
    auto input_column = column_map.getStringColumn(_input_column_name);
    uint32_t num_rows = column_map.numRows();

    ColumnPtr output_column;
    if (_output_range) {
      output_column = createSparseArrayColumn()
    } else {
      output_column = createTokenArrayColumn();
    }
    column_map.setColumn(_output_column_name, output_column);

#pragma omp parallel for default(none) shared(num_rows, column, _output_range)
    for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
      std::string text = (*column)[row_idx];
      std::vector<uint32_t> unigrams =
          TextEncodingUtils::computeRawUnigramsWithRange(text, _output_range);

      TextEncodingUtils::sumRepeatedIndices(
          unigrams, /* base_value= */ 1.0, [&](uint32_t unigram, float value) {
            vec.addSparseFeatureToSegment(unigram, value);
          });
    }
  }

  std::string _input_column_name;
  std::string _output_column_name;
  std::optional<uint32_t> _output_range;
};

}  // namespace thirdai::dataset