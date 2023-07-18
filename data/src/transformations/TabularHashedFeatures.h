#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/MurmurHash.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <string>
#include <vector>

namespace thirdai::data {

/**
 * @brief This column assumes as input N SparseValueColumns, computes either
 * unigrams or pairgrams across all the indices and returns the results as a new
 * IndexValueArrayColumn. Each input value will be salted according to its
 * column of origin to limit duplicate values all hashing to the same location.
 */
class TabularHashedFeatures : public Transformation {
 public:
  TabularHashedFeatures(std::vector<std::string> input_column_names,
                        std::string output_column_name, uint32_t output_range,
                        bool use_pairgrams = false)
      : _input_column_names(std::move(input_column_names)),
        _output_column_name(std::move(output_column_name)),
        _output_range(output_range),
        _use_pairgrams(use_pairgrams) {}

  ColumnMap apply(ColumnMap column_map) const final {
    std::vector<ValueColumnPtr<uint32_t>> columns;
    // we hash the name of each column here so we can combine hashes later on
    // and have unique values across columns
    std::vector<uint32_t> column_name_hashes;
    for (const auto& col_name : _input_column_names) {
      columns.push_back(column_map.getValueColumn<uint32_t>(col_name));
      column_name_hashes.push_back(dataset::token_encoding::seededMurmurHash(
          /* key = */ col_name.c_str(), /* len = */ col_name.size()));
    }

    uint32_t num_rows = column_map.numRows();
    std::vector<std::vector<uint32_t>> tabular_hash_values(num_rows);
#pragma omp parallel for default(none) \
    shared(num_rows, columns, column_name_hashes, tabular_hash_values)
    for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
      std::vector<uint32_t> salted_unigrams;
      uint32_t col_num = 0;
      for (const ValueColumnPtr<uint32_t>& column : columns) {
        // TODO(david): it may be unnecessary to hash again but technically the
        // original uint32_t values may not be from the correct universal hash
        // distribution. We cast the uint32_t to char* so we can use murmur hash
        const char* val_to_hash =
            reinterpret_cast<const char*>(&(column->value(row_idx)));
        uint32_t hashed_col_val = dataset::token_encoding::seededMurmurHash(
            val_to_hash, /* len = */ 4);
        // to avoid two identical values in different columns from having the
        // same hash value we combine the with the hash of the column name of
        // origin
        salted_unigrams.push_back(hashing::combineHashes(
            hashed_col_val, column_name_hashes[col_num]));
        col_num++;
      }

      if (_use_pairgrams) {
        // we don't deduplicate pairgrams since we ensure unique hash values
        // above, thus reducing the chance of duplicates.
        std::vector<uint32_t> row_pairgrams =
            dataset::token_encoding::pairgrams(salted_unigrams);
        dataset::token_encoding::mod(row_pairgrams, _output_range);
        tabular_hash_values[row_idx] = row_pairgrams;
      } else {
        for (uint32_t& salted_unigram : salted_unigrams) {
          salted_unigram = salted_unigram % _output_range;
        }
        tabular_hash_values[row_idx] = salted_unigrams;
      }
    }

    auto output_column =
        TokenArrayColumn::make(std::move(tabular_hash_values), _output_range);

    column_map.setColumn(_output_column_name, output_column);

    return column_map;
  }

 private:
  // Private constructor for cereal.
  TabularHashedFeatures()
      : _input_column_names(),
        _output_column_name(),
        _output_range(0),
        _use_pairgrams(false) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Transformation>(this), _input_column_names,
            _output_column_name, _output_range, _use_pairgrams);
  }

  std::vector<std::string> _input_column_names;
  std::string _output_column_name;
  uint32_t _output_range;
  bool _use_pairgrams;
};

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::TabularHashedFeatures)
