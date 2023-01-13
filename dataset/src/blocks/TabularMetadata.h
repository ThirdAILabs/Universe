#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <dataset/src/BatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::dataset {

enum class TabularDataType {
  Numeric,
  Categorical,
  Label,
  Ignore,
};  // TODO(david) add datetime/text support

/**
 * @brief This object stores metadata about a Tabular Dataset like column
 * type information, max/min values for numeric columns, etc.
 *
 * @param column_dtypes Types of the observed columns.
 * @param col_min_maxes Mapping from column number to a pair of min/max values.
 * Must specify a pair for each Numeric type as specified in column_dtypes.
 * @param class_name_to_id ThreadSafeVocabularyPtr of the labels. Should be
 * fixed
 * @param column_names Vector of column names. Must match the length of
 * column_dtypes. Required for use in RCA but optional for general use.
 * @param col_to_num_bins Optional mapping from column number to the number of
 * bins to use for that column. Must specify a pair for each Numeric type as
 * specified in column_dtypes. If map is not passed in, TabularHashFeatures
 * default the number of bins to 10.
 *
 * TODO(david): look at a TabularBinningStrategy class to try out different
 * methods?
 */
class TabularMetadata {
 public:
  TabularMetadata(
      std::vector<TabularDataType> column_dtypes,
      std::unordered_map<uint32_t, std::pair<double, double>> col_min_maxes,
      ThreadSafeVocabularyPtr class_name_to_id,
      const std::vector<std::string>& column_names = {},
      std::optional<std::unordered_map<uint32_t, uint32_t>> col_to_num_bins =
          std::nullopt)
      : _column_dtypes(std::move(column_dtypes)),
        _col_min_maxes(std::move(col_min_maxes)),
        _class_name_to_id(std::move(class_name_to_id)),
        _col_to_num_bins(std::move(col_to_num_bins)) {
    if (column_names.empty()) {
      for (uint32_t col = 0; col < _column_dtypes.size(); col++) {
        _column_identifiers.push_back(col);
      }
    } else {
      for (const auto& name : column_names) {
        _column_identifiers.push_back(name);
      }
    }

    if (!_column_identifiers.empty() &&
        _column_identifiers.size() != _column_dtypes.size()) {
      throw std::invalid_argument(
          "A non-empty column_names input should have the same number of "
          "elements as the column_dtypes input.");
    }
  }

  std::vector<ColumnIdentifier>& concreteBlockColumnIdentifiers() {
    return _column_identifiers;
  }

  ThreadSafeVocabularyPtr getClassToIdMap() { return _class_name_to_id; }

  uint32_t numColumns() const { return _column_dtypes.size(); }

  TabularDataType colType(uint32_t col) { return _column_dtypes[col]; }

  static uint32_t getStringHashValue(const std::string& str_val, uint32_t col) {
    // we salt the hash to ensure hashes are unique across columns
    const char* char_salt = reinterpret_cast<const char*>(&col);
    std::string str_salt(char_salt, 4);
    std::string unique_category = str_val + str_salt;
    return TextEncodingUtils::computeUnigram(unique_category.data(),
                                             unique_category.size());
  }

  uint32_t getNumericHashValue(uint32_t col, const std::string& str_val,
                               std::exception_ptr& exception_ptr) {
    uint32_t bin = getColBin(col, str_val, exception_ptr);
    // to ensure hashes are unique across columns we add salt based on the col
    uint64_t uniqueBin =
        static_cast<uint64_t>(bin) << 32 | static_cast<uint64_t>(col);
    const char* val_to_hash = reinterpret_cast<const char*>(&uniqueBin);
    return TextEncodingUtils::computeUnigram(val_to_hash, /* len = */ 8);
  }

  ColumnIdentifier columnIdentifier(uint32_t col) {
    return _column_identifiers.at(col);
  }

 private:
  double getColBinsize(uint32_t col) {
    return (colMax(col) - colMin(col)) / numBins(col);
  }

  /**
   * For values between the min and the max (inclusive), we divide that range
   * into N uniform chunks and return a bin number from 0 to N-1. Additionally
   * we support three special cases of bins:
   *    - if we're given an empty str_val we return bin number N
   *    - if the str_val is less than the min we return bin number 0
   *    - if the str_val is greater than the max we return bin number N - 1
   */
  uint32_t getColBin(uint32_t col, const std::string& str_val,
                     std::exception_ptr& exception_ptr) {
    if (str_val.empty()) {
      return numBins(col);
    }
    double value;
    try {
      value = std::stod(str_val);
    } catch (std::invalid_argument& e) {
      exception_ptr = std::make_exception_ptr(std::invalid_argument(
          "Could not process column " + std::to_string(col) +
          " as type 'numeric.' Received value: '" + str_val + ".'"));

      // Since we have set the block exception above, the program will
      // fail once all threads finish. Since we can't throw an exception within
      // a pragma thread, we just have to keep the program running until then.
      // Thus we return some arbitrary value to do that.
      return 0;
    }
    uint32_t min = colMin(col);
    uint32_t max = colMax(col);
    if (value < min) {
      return 0;
    }
    if (value > max) {
      return numBins(col) - 1;
    }

    double binsize = getColBinsize(col);
    if (binsize == 0) {
      return 0;
    }
    return static_cast<uint32_t>(std::round((value - colMin(col)) / binsize));
  }

  double colMin(uint32_t col) { return _col_min_maxes.at(col).first; }

  double colMax(uint32_t col) { return _col_min_maxes.at(col).second; }

  static constexpr uint32_t DEFAULT_NUM_BINS = 10;

  uint32_t numBins(uint32_t col) {
    if (_col_to_num_bins) {
      return (*_col_to_num_bins).at(col);
    }
    return DEFAULT_NUM_BINS;
  }

  void verifyInputs() {
    for (uint32_t col = 0; col < _column_dtypes.size(); col++) {
      if (colType(col) == TabularDataType::Numeric) {
        if (!_col_min_maxes.count(col)) {
          throw std::invalid_argument(
              "Column " + std::to_string(col) +
              " specified as Numeric has no given min/max values.");
        }
        if (_col_to_num_bins && !_col_to_num_bins->count(col)) {
          throw std::invalid_argument(
              "Column " + std::to_string(col) +
              " specified as Numeric has no given number of bins.");
        }
      }
      if (colType(col) != TabularDataType::Numeric) {
        if (_col_min_maxes.count(col)) {
          throw std::invalid_argument(
              "Column " + std::to_string(col) +
              " specified as non-numeric has min/max values.");
        }
        if (_col_to_num_bins && _col_to_num_bins->count(col)) {
          throw std::invalid_argument(
              "Column " + std::to_string(col) +
              " specified as non-numeric has bin values.");
        }
      }
      if (colType(col) == TabularDataType::Label) {
        if (_label_col) {
          throw std::invalid_argument(
              "Found multiple 'label' columns in dataset.");
        }
        _label_col = col;
      }
    }
  }

  // Private constructor for cereal
  TabularMetadata() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_column_dtypes, _col_min_maxes, _column_identifiers,
            _col_to_num_bins, _class_name_to_id, _label_col);
  }

  std::vector<TabularDataType> _column_dtypes;
  std::unordered_map<uint32_t, std::pair<double, double>> _col_min_maxes;

  ThreadSafeVocabularyPtr _class_name_to_id;

  std::vector<ColumnIdentifier> _column_identifiers;

  // three additional bins are reserved: one for empty values, one for values
  // less than the min, and one for values greater than the max
  std::optional<std::unordered_map<uint32_t, uint32_t>> _col_to_num_bins;

  std::optional<uint32_t> _label_col;
};

using TabularMetadataPtr = std::shared_ptr<TabularMetadata>;

}  // namespace thirdai::dataset
