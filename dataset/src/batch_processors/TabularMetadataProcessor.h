#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include "ProcessorUtils.h"
#include <dataset/src/BatchProcessor.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <cmath>
#include <limits>

namespace thirdai::dataset {

enum class TabularDataType {
  Numeric,
  Categorical,
  Label
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
 * specified in column_dtypes. If map is not passed in, TabularPairGram default
 * the number of bins to 10.
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
      std::vector<std::string> column_names = {},
      std::optional<std::unordered_map<uint32_t, uint32_t>> col_to_num_bins =
          std::nullopt)
      : _column_dtypes(std::move(column_dtypes)),
        _col_min_maxes(std::move(col_min_maxes)),
        _class_name_to_id(std::move(class_name_to_id)),
        _column_names(std::move(column_names)),
        _col_to_num_bins(std::move(col_to_num_bins)) {
    // column names are for future RCA module but we aren't necessary for
    // anything else. Here we check that if we provide column_names that they
    // are the same length as column_dtypes
    if (!_column_names.empty() &&
        _column_names.size() != _column_dtypes.size()) {
      throw std::invalid_argument(
          "A non-empty column_names input should have the same number of "
          "elements as the column_dtypes input.");
    }
    verifyInputs();
  }

  uint32_t getLabelCol() const { return *_label_col; }

  ThreadSafeVocabularyPtr getClassToIdMap() { return _class_name_to_id; }

  uint32_t numColumns() const { return _column_dtypes.size(); }

  TabularDataType colType(uint32_t col) { return _column_dtypes[col]; }

  uint32_t getStringHashValue(const std::string& str_val, uint32_t col) const {
    // to ensure hashes are unique across columns we add salt based on the col
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

 private:
  double getColBinsize(uint32_t col) {
    return (colMax(col) - colMin(col)) / numBins(col);
  }

  uint32_t getColBin(uint32_t col, const std::string& str_val,
                     std::exception_ptr& exception_ptr) {
    // map empty values to their own bin
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
      switch (colType(col)) {
        case TabularDataType::Numeric: {
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
          break;
        }
        case TabularDataType::Categorical: {
          if (_col_min_maxes.count(col)) {
            throw std::invalid_argument(
                "Column " + std::to_string(col) +
                " specified as Categorical has min/max values.");
          }
          if (_col_to_num_bins && _col_to_num_bins->count(col)) {
            throw std::invalid_argument(
                "Column " + std::to_string(col) +
                " specified as Categorical has bin values.");
          }
          break;
        }
        case TabularDataType::Label: {
          if (_label_col) {
            throw std::invalid_argument(
                "Found multiple 'label' columns in dataset.");
          }
          _label_col = col;
          break;
        }
      }
    }
  }

  // Private constructor for cereal
  TabularMetadata() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_column_dtypes, _col_min_maxes, _column_names, _col_to_num_bins,
            _class_name_to_id, _label_col);
  }

  std::vector<TabularDataType> _column_dtypes;
  std::unordered_map<uint32_t, std::pair<double, double>> _col_min_maxes;

  ThreadSafeVocabularyPtr _class_name_to_id;

  std::vector<std::string> _column_names;

  // one additional bin is reserved for empty values
  std::optional<std::unordered_map<uint32_t, uint32_t>> _col_to_num_bins;

  std::optional<uint32_t> _label_col;
};

using TabularMetadataPtr = std::shared_ptr<TabularMetadata>;

class TabularMetadataProcessor : public ComputeBatchProcessor {
 public:
  explicit TabularMetadataProcessor(std::vector<std::string> column_types,
                                    uint32_t n_classes)
      : _delimiter(','),
        _label_col(std::nullopt),
        _class_name_to_id(
            ThreadSafeVocabulary::make(/* vocab_size = */ n_classes)) {
    for (uint32_t col_idx = 0; col_idx < column_types.size(); col_idx++) {
      std::string col_type = column_types[col_idx];
      if (col_type == "label") {
        if (_label_col) {
          throw std::invalid_argument(
              "Found multiple 'label' columns in dataset.");
        }
        _label_col = col_idx;
        _column_dtypes.push_back(TabularDataType::Label);
      } else if (col_type == "categorical") {
        _column_dtypes.push_back(TabularDataType::Categorical);
      } else if (col_type == "numeric") {
        _column_dtypes.push_back(TabularDataType::Numeric);
        _col_min_maxes[col_idx] =
            std::make_pair(std::numeric_limits<double>::min(),
                           std::numeric_limits<double>::max());
      } else {
        throw std::invalid_argument(
            "Received datatype '" + col_type +
            "' expected one of 'label', 'categorical', or 'numeric'.");
      }
    }
    if (!_label_col) {
      throw std::invalid_argument("Dataset does not contain a 'label' column.");
    }
  }

  bool expectsHeader() const final { return true; }

  void processHeader(const std::string& header) final {
    std::vector<std::string_view> column_names =
        ProcessorUtils::parseCsvRow(header, _delimiter);
    verifyNumColumns(column_names);
    for (auto col_name : column_names) {
      _column_names.push_back(std::string(col_name));
    }
  }

  void processRow(const std::string& row) final {
    std::vector<std::string_view> values =
        ProcessorUtils::parseCsvRow(row, _delimiter);
    verifyNumColumns(values);
    for (uint32_t col = 0; col < values.size(); col++) {
      std::string str_value(values[col]);
      switch (_column_dtypes[col]) {
        case TabularDataType::Numeric: {
          processNumeric(str_value, col);
          break;
        }
        case TabularDataType::Categorical:
          break;
        case TabularDataType::Label: {
          _class_name_to_id->getUid(str_value);
          break;
        }
      }
    }
  }

  TabularMetadataPtr getTabularMetadata() {
    _class_name_to_id->fixVocab();
    return std::make_shared<TabularMetadata>(_column_dtypes, _col_min_maxes,
                                             _class_name_to_id, _column_names);
  }

 private:
  void processNumeric(const std::string& str_value, uint32_t col) {
    // TODO(david) handle nan/inf/large number cases

    // we want to process empty values and put them in their own bin later, thus
    // we don't fail here
    if (str_value.empty()) {
      return;
    }
    try {
      double val = std::stod(str_value);
      if (val < _col_min_maxes[col].first) {
        _col_min_maxes[col].first = val;
      }
      if (val > _col_min_maxes[col].second) {
        _col_min_maxes[col].second = val;
      }
    } catch (std::invalid_argument& e) {
      throw std::invalid_argument(
          "Could not process column " + std::to_string(col) +
          " as type 'numeric.' Received value: '" + str_value + ".'");
    }
  }

  void verifyNumColumns(const std::vector<std::string_view>& values) {
    if (values.size() != _column_dtypes.size()) {
      throw std::invalid_argument("Csv format error. Expected " +
                                  std::to_string(_column_dtypes.size()) +
                                  " columns but received " +
                                  std::to_string(values.size()) + " columns.");
    }
  }

  char _delimiter;
  std::optional<uint32_t> _label_col;
  std::vector<TabularDataType> _column_dtypes;
  std::unordered_map<uint32_t, std::pair<double, double>> _col_min_maxes;
  std::vector<std::string> _column_names;
  ThreadSafeVocabularyPtr _class_name_to_id;
};

}  // namespace thirdai::dataset
