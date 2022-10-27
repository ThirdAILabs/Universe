#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
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
 * @brief This column assumes as input N SparseValueColumns, computes pairgrams
 * across all the indices and returns the results as a new
 * IndexValueArrayColumn. Each input value will be salted according to its
 * column of origin to limit duplicate values all hashing to the same location.
 */
class CrossColumnPairgram : public Transformation {
 public:
  CrossColumnPairgram(std::vector<std::string> input_column_names,
                      std::string output_column_name, uint32_t output_range)
      : _input_column_names(std::move(input_column_names)),
        _output_column_name(std::move(output_column_name)),
        _output_range(output_range) {}

  void apply(ColumnMap& column_map) final {
    std::vector<SparseValueColumnPtr> columns;
    // we hash the name of each column here so we can combine hashes later on
    // and have unique values across columns
    std::vector<uint32_t> column_name_hashes(_input_column_names.size());
    for (const auto& col_name : _input_column_names) {
      columns.push_back(column_map.getSparseValueColumn(col_name));
      column_name_hashes.push_back(TextEncodingUtils::computeUnigram(
          /* key = */ col_name.c_str(), /* len = */ col_name.size()));
    }

    uint32_t num_rows = column_map.numRows();
    std::vector<std::vector<uint32_t>> pairgrams(num_rows);

    // set here because pragma doesnt like sharing private member variables
    uint32_t output_range = _output_range;
#pragma omp parallel for default(none) \
    shared(num_rows, columns, column_name_hashes, pairgrams, output_range)
    for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
      std::vector<uint32_t> salted_unigrams;
      uint32_t col_num = 0;
      for (const SparseValueColumnPtr& column : columns) {
        // TODO(david): it may be unnecessary to hash again but technically the
        // original uint32_t values may not be from the correct universal hash
        // distribution. We cast the uint32_t to char* so we can use murmur hash
        const char* val_to_hash =
            reinterpret_cast<const char*>(&((*column)[row_idx]));
        uint32_t hashed_col_val =
            TextEncodingUtils::computeUnigram(val_to_hash, /* len = */ 4);
        // to avoid two identical values in different columns from having the
        // same hash value we combine the with the hash of the column name of
        // origin
        salted_unigrams.push_back(hashing::HashUtils::combineHashes(
            hashed_col_val, column_name_hashes[col_num]));
        col_num++;
      }

      // we don't deduplicate pairgrams since we ensure unique hash values
      // above, thus reducing the chance of duplicates.
      std::vector<uint32_t> row_pairgrams =
          TextEncodingUtils::computeRawPairgramsFromUnigrams(salted_unigrams,
                                                             output_range);
      pairgrams[row_idx] = row_pairgrams;
    }

    auto output_column = std::make_shared<VectorSparseArrayColumn>(
        std::move(pairgrams), _output_range);
    column_map.setColumn(_output_column_name, output_column);
  }

 private:
  // Private constructor for cereal.
  CrossColumnPairgram()
      : _input_column_names(), _output_column_name(), _output_range(0) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Transformation>(this), _input_column_names,
            _output_column_name, _output_range);
  }

  std::vector<std::string> _input_column_names;
  std::string _output_column_name;
  uint32_t _output_range;
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::CrossColumnPairgram)
