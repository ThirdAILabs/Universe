#pragma once

#include <hashing/src/MurmurHash.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
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
    // we hash the name of each column here sowe can combine hashes later on
    // and have unique values across columns
    std::vector<uint32_t> column_hashes(_input_column_names.size());
    for (const auto& col_name : _input_column_names) {
      columns.push_back(column_map.getSparseValueColumn(col_name));
      column_hashes.push_back(
          TextEncodingUtils::computeUnigram(col_name.c_str(), col_name.size()));
    }

    uint32_t num_rows = column_map.numRows();
    std::vector<std::vector<uint32_t>> column_values(num_rows);

#pragma omp parallel for default(none) \
    shared(num_rows, columns, column_hashes, column_values, _output_range)
    for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
      std::vector<uint32_t> unigram_hashes(columns.size());
      uint32_t col_num = 0;
      for (const auto& column : columns) {
        const char* val_to_hash =
            reinterpret_cast<const char*>(&(*column)[row_idx]);
        uint32_t hashed_col_val =
            TextEncodingUtils::computeUnigram(val_to_hash, /* len = */ 4);
        // to avoid two identical values in different columns from having the
        // same hash value we combine the with the hash of the column name of
        // origin
        unigram_hashes.push_back(hashing::HashUtils::combineHashes(
            hashed_col_val, column_hashes[col_num]));
        col_num++;
      }

      // we don't deduplicate pairgrams since we ensure unique hash values
      // above, thus reducing the change of duplicates
      std::vector<uint32_t> pairgrams =
          TextEncodingUtils::computeRawPairgramsFromUnigrams(unigram_hashes,
                                                             _output_range);
      column_values[row_idx] = pairgrams;
    }

    auto output_column =
        std::make_shared<VectorArrayColumn<uint32_t>>(std::move(column_values));
    column_map.setColumn(_output_column_name, output_column);
  }

 private:
  std::vector<std::string> _input_column_names;
  std::string _output_column_name;
  uint32_t _output_range;
};

}  // namespace thirdai::dataset