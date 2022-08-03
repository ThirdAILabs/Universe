#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include "ProcessorUtils.h"
#include <dataset/src/BatchProcessor.h>
#include <dataset/src/encodings/text/TextEncodingUtils.h>
#include <cmath>
#include <limits>

namespace thirdai::dataset {

enum class TabularDataType {
  Numeric,
  Categorical,
  Label
};  // TODO(david) add datetime/text support

class TabularMetadata {
 public:
  /**
   * This object stores metadata about a Tabular Dataset like column type
   * information, max/min values for numeric columns, etc.
   */
  TabularMetadata(uint32_t label_col_index,
                  std::vector<TabularDataType> column_dtypes,
                  std::unordered_map<uint32_t, double> col_to_max_val,
                  std::unordered_map<uint32_t, double> col_to_min_val,
                  uint32_t max_salt_len)
      : _num_non_empty_bins(10),
        _label_col_index(label_col_index),
        _max_salt_len(max_salt_len),
        _column_dtypes(std::move(column_dtypes)),
        _col_to_max_val(std::move(col_to_max_val)),
        _col_to_min_val(std::move(col_to_min_val)) {}

  void addClass(const std::string& class_name) {
    uint32_t label = numClasses();
    _class_to_class_id[class_name] = label;
    _class_id_to_class.push_back(class_name);
  }

  void updateColMaxMin(uint32_t col, double val) {
    if (_col_to_max_val[col] < val) {
      _col_to_max_val[col] = val;
    }
    if (_col_to_min_val[col] > val) {
      _col_to_min_val[col] = val;
    }
  }

  void setColumnNames(
      std::unordered_map<uint32_t, std::string> col_to_col_name) {
    _col_to_col_name = col_to_col_name;
  }

  uint32_t numColumns() const { return _column_dtypes.size(); }

  TabularDataType getColType(uint32_t col) const { return _column_dtypes[col]; }

  uint32_t numClasses() const { return _class_id_to_class.size(); }

  bool containsClassName(const std::string& str_value) const {
    return _class_to_class_id.count(str_value);
  }

  uint32_t getLabelCol() const { return _label_col_index; }

  std::vector<std::string> getClassIdToNames() { return _class_id_to_class; }

  std::unordered_map<std::string, uint32_t> getClassToIdMap() {
    return _class_to_class_id;
  }

  uint32_t getStringHashValue(const std::string& str_val, uint32_t col) const {
    std::string unique_category = str_val + getColSalt(col);
    return TextEncodingUtils::computeUnigram(unique_category.data(),
                                             unique_category.size());
  }

  uint32_t getNumericHashValue(uint32_t col, const std::string& str_val,
                               std::exception_ptr& exception_ptr) {
    uint32_t bin = getColBin(col, str_val, exception_ptr);
    uint64_t uniqueBin =
        static_cast<uint64_t>(bin) << 32 | static_cast<uint64_t>(col);
    const char* val_to_hash = reinterpret_cast<const char*>(&uniqueBin);
    return TextEncodingUtils::computeUnigram(val_to_hash, /* len = */ 8);
  }

 private:
  double getColMin(uint32_t col) { return _col_to_min_val[col]; }

  double getColBinsize(uint32_t col) {
    return (_col_to_max_val[col] - _col_to_min_val[col]) / _num_non_empty_bins;
  }

  uint32_t getColBin(uint32_t col, const std::string& str_val,
                     std::exception_ptr& exception_ptr) {
    // map empty values to their own bin
    if (str_val.empty()) {
      return _num_non_empty_bins;
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
    return static_cast<uint32_t>(
        std::round((value - getColMin(col)) / getColBinsize(col)));
  }

  /**
   * To compute pairgrams of categories across columns we need all categories to
   * be unique. This method returns "salt" to make each category unique
   * dependent on its column. By adding the column number padded to a fixed
   * length, we ensure this characteristic.
   */
  std::string getColSalt(uint32_t col) const {
    std::string col_str = std::to_string(col);
    if (col_str.size() < _max_salt_len) {
      col_str.insert(0, _max_salt_len - col_str.size(), '0');
    }
    return col_str;
  }

  // Private constructor for cereal
  TabularMetadata() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_num_non_empty_bins, _label_col_index, _column_dtypes,
            _col_to_max_val, _col_to_min_val, _class_to_class_id,
            _class_id_to_class, _col_to_col_name);
  }

  // one additional bin is reserved for empty values
  uint32_t _num_non_empty_bins;
  uint32_t _label_col_index;
  uint32_t _max_salt_len;
  std::vector<TabularDataType> _column_dtypes;
  std::unordered_map<uint32_t, double> _col_to_max_val;
  std::unordered_map<uint32_t, double> _col_to_min_val;
  std::unordered_map<std::string, uint32_t> _class_to_class_id;
  std::vector<std::string> _class_id_to_class;
  std::unordered_map<uint32_t, std::string> _col_to_col_name;
};

class TabularMetadataProcessor : public ComputeBatchProcessor {
 public:
  explicit TabularMetadataProcessor(std::vector<std::string> given_column_types,
                                    uint32_t n_classes)
      : _delimiter(','), _n_classes(n_classes) {
    bool found_label_column = false;
    uint32_t label_col_index = 0;
    std::vector<TabularDataType> column_dtypes;
    std::unordered_map<uint32_t, double> col_to_max_val;
    std::unordered_map<uint32_t, double> col_to_min_val;

    for (uint32_t col_idx = 0; col_idx < given_column_types.size(); col_idx++) {
      if (given_column_types[col_idx] == "label" && found_label_column) {
        throw std::invalid_argument(
            "Found multiple 'label' columns in dataset.");
      }

      if (given_column_types[col_idx] == "label") {
        found_label_column = true;
        label_col_index = col_idx;
        column_dtypes.push_back(TabularDataType::Label);

      } else if (given_column_types[col_idx] == "categorical") {
        column_dtypes.push_back(TabularDataType::Categorical);

      } else if (given_column_types[col_idx] == "numeric") {
        column_dtypes.push_back(TabularDataType::Numeric);
        col_to_max_val[col_idx] = std::numeric_limits<double>::min();
        col_to_min_val[col_idx] = std::numeric_limits<double>::max();

      } else {
        throw std::invalid_argument(
            "Received datatype '" + given_column_types[col_idx] +
            "' expected one of 'label', 'categorical', or 'numeric'.");
      }
    }
    if (!found_label_column) {
      throw std::invalid_argument("Dataset does not contain a 'label' column.");
    }
    uint32_t max_salt_len = std::to_string(given_column_types.size()).size();

    _metadata = std::make_shared<TabularMetadata>(label_col_index,
                                                  column_dtypes, col_to_max_val,
                                                  col_to_min_val, max_salt_len);
  }

  bool expectsHeader() const final { return true; }

  void processHeader(const std::string& header) final {
    std::vector<std::string_view> column_names =
        ProcessorUtils::parseCsvRow(header, _delimiter);
    verifyNumColumns(column_names);
    std::unordered_map<uint32_t, std::string> col_to_col_name;
    for (uint32_t col = 0; col < column_names.size(); col++) {
      col_to_col_name[col] = column_names[col];
    }
    _metadata->setColumnNames(col_to_col_name);
  }

  void processRow(const std::string& row) final {
    std::vector<std::string_view> values =
        ProcessorUtils::parseCsvRow(row, _delimiter);
    verifyNumColumns(values);
    for (uint32_t col = 0; col < _metadata->numColumns(); col++) {
      std::string str_value(values[col]);
      switch (_metadata->getColType(col)) {
        case TabularDataType::Numeric: {
          processNumeric(str_value, col);
          break;
        }
        case TabularDataType::Categorical:
          break;
        case TabularDataType::Label: {
          processLabel(str_value);
          break;
        }
      }
    }
  }

  std::shared_ptr<TabularMetadata> getMetadata() { return _metadata; }

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
      _metadata->updateColMaxMin(col, val);
    } catch (std::invalid_argument& e) {
      throw std::invalid_argument(
          "Could not process column " + std::to_string(col) +
          " as type 'numeric.' Received value: '" + str_value + ".'");
    }
  }

  void processLabel(const std::string& str_value) {
    if (!_metadata->containsClassName(str_value)) {
      if (_metadata->numClasses() == _n_classes) {
        throw std::invalid_argument(
            "Expected " + std::to_string(_n_classes) +
            " classes but found an additional class: '" + str_value + ".'");
      }
      _metadata->addClass(str_value);
    }
  }

  void verifyNumColumns(const std::vector<std::string_view>& values) {
    if (values.size() != _metadata->numColumns()) {
      throw std::invalid_argument("Csv format error. Expected " +
                                  std::to_string(_metadata->numColumns()) +
                                  " columns but received " +
                                  std::to_string(values.size()) + " columns.");
    }
  }

  char _delimiter;
  uint32_t _n_classes;
  std::shared_ptr<TabularMetadata> _metadata;
};

}  // namespace thirdai::dataset
