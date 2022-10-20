#pragma once

#include <dataset/src/data_pipeline/Column.h>
#include <dataset/src/data_pipeline/ColumnMap.h>
#include <dataset/src/data_pipeline/Transformation.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <string>
#include <vector>

namespace thirdai::dataset {

class CrossColumnPairgram : public Transformation {
  CrossColumnPairgram(std::vector<std::string> input_column_names,
                      std::string output_column_name, uint32_t output_range)
      : _input_column_names(std::move(input_column_names)),
        _output_column_name(std::move(output_column_name)),
        _output_range(output_range) {}

  void apply(ColumnMap& column_map) final {
    std::vector<std::shared_ptr<SparseValueColumn>> columns(
        _input_column_names.size());
    for (const auto& col_name : _input_column_names) {
      columns.push_back(column_map.getSparseValueColumn(col_name));
    }

#pragma omp parallel for default(none) \
    shared(column_map, columns, _output_range)
    for (uint32_t row_idx = 0; row_idx < column_map.numRows(); row_idx++) {
      std::vector<uint32_t> unigram_hashes(columns.size());
      uint32_t col_num = 0;
      for (const auto& column : columns) {
        // to avoid two identical values in different columns from having the
        // same hash value we add the column number as "salt" to make it unique
        // we use a 64 bit int, the first 32 bits being the value and the next
        // 32 bits being the column number
        uint64_t salted_value = static_cast<uint64_t>((*column)[row_idx])
                                    << 32 |
                                static_cast<uint64_t>(col_num);
        const char* val_to_hash = reinterpret_cast<const char*>(&salted_value);
        unigram_hashes.push_back(
            TextEncodingUtils::computeUnigram(val_to_hash, /* len = */ 8));
        col_num++;
      }

      std::vector<uint32_t> pairgram_hashes =
          TextEncodingUtils::computeRawPairgramsFromUnigrams(unigram_hashes,
                                                             _output_range);

      std::vector<uint32_t> pairgrams(pairgram_hashes.size());
      TextEncodingUtils::sumRepeatedIndices(
          pairgram_hashes, /* base_value = */ 1.0,
          [&](uint32_t pairgram, float value) {
            (void)value;
            pairgrams.push_back(pairgram);
          });
    }
  }

  std::vector<std::string> _input_column_names;
  std::string _output_column_name;
  uint32_t _output_range;
};

}  // namespace thirdai::dataset