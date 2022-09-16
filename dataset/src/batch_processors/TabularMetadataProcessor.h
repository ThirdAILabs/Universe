#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include "ProcessorUtils.h"
#include <dataset/src/BatchProcessor.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <cmath>
#include <limits>

namespace thirdai::dataset {

enum class TabularDataType {
  Numeric,
  Categorical,
  Label
};  // TODO(david) add datetime/text support

/**
 * This object stores metadata about a Tabular Dataset like column type
 * information, max/min values for numeric columns, etc.
 */
class TabularMetadata {
 public:
  TabularMetadata(std::vector<TabularDataType> column_dtypes,
                  std::unordered_map<uint32_t, double> col_to_max_val,
                  std::unordered_map<uint32_t, double> col_to_min_val,
                  std::unordered_map<std::string, uint32_t> class_to_class_id,
                  std::optional<std::unordered_map<uint32_t, uint32_t>>
                      col_to_num_bins = std::nullopt)
      : _column_dtypes(std::move(column_dtypes)),
        _col_to_max_val(std::move(col_to_max_val)),
        _col_to_min_val(std::move(col_to_min_val)),
        _class_to_class_id(std::move(class_to_class_id)),
        _col_to_num_bins(std::move(col_to_num_bins)) {
    auto itr = std::find(_column_dtypes.begin(), _column_dtypes.end(),
                         TabularDataType::Label);
    if (itr != _column_dtypes.end()) {
      _label_col_index = std::distance(_column_dtypes.begin(), itr);
    } else {
      throw std::invalid_argument("No label col passed in.");
    }
  }

  uint32_t getLabelCol() const { return _label_col_index; }

  uint32_t numColumns() const { return _column_dtypes.size(); }

  std::unordered_map<std::string, uint32_t> getClassToIdMap() {
    return _class_to_class_id;
  }

  TabularDataType colType(uint32_t col) { return _column_dtypes[col]; }

  double colMax(uint32_t col) { return _col_to_max_val[col]; }

  double colMin(uint32_t col) { return _col_to_min_val[col]; }

  static constexpr uint32_t DEFAULT_NUM_BINS = 10;

  uint32_t numBins(uint32_t col) {
    if (_col_to_num_bins) {
      return (*_col_to_num_bins)[col];
    }
    return DEFAULT_NUM_BINS;
  }

 private:
  // Private constructor for cereal
  TabularMetadata() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_column_dtypes, _col_to_max_val, _col_to_min_val,
            _class_to_class_id, _col_to_num_bins, _label_col_index);
  }

  std::vector<TabularDataType> _column_dtypes;
  std::unordered_map<uint32_t, double> _col_to_max_val;
  std::unordered_map<uint32_t, double> _col_to_min_val;
  std::unordered_map<std::string, uint32_t> _class_to_class_id;

  // one additional bin is reserved for empty values
  std::optional<std::unordered_map<uint32_t, uint32_t>> _col_to_num_bins;

  uint32_t _label_col_index;
};

class TabularMetadataProcessor : public ComputeBatchProcessor {
 public:
  explicit TabularMetadataProcessor(std::vector<std::string> column_types,
                                    uint32_t n_classes)
      : _delimiter(','), _n_classes(n_classes) {
    bool found_label_column = false;

    for (uint32_t col_idx = 0; col_idx < column_types.size(); col_idx++) {
      std::string col_type = column_types[col_idx];
      if (col_type == "label") {
        if (found_label_column) {
          throw std::invalid_argument(
              "Found multiple 'label' columns in dataset.");
        }
        found_label_column = true;
        _column_dtypes.push_back(TabularDataType::Label);
      } else if (col_type == "categorical") {
        _column_dtypes.push_back(TabularDataType::Categorical);
      } else if (col_type == "numeric") {
        _column_dtypes.push_back(TabularDataType::Numeric);
        _col_to_max_val[col_idx] = std::numeric_limits<double>::min();
        _col_to_min_val[col_idx] = std::numeric_limits<double>::max();
      } else {
        throw std::invalid_argument(
            "Received datatype '" + col_type +
            "' expected one of 'label', 'categorical', or 'numeric'.");
      }
    }
    if (!found_label_column) {
      throw std::invalid_argument("Dataset does not contain a 'label' column.");
    }
  }

  bool expectsHeader() const final { return true; }

  void processHeader(const std::string& header) final {
    std::vector<std::string_view> column_names =
        ProcessorUtils::parseCsvRow(header, _delimiter);
    verifyNumColumns(column_names);
    std::unordered_map<uint32_t, std::string> col_to_col_name;
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
          processLabel(str_value);
          break;
        }
      }
    }
  }

  std::shared_ptr<TabularMetadata> getMetadata() {
    return std::make_shared<TabularMetadata>(
        _column_dtypes, _col_to_max_val, _col_to_min_val, _class_to_class_id);
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
      if (_col_to_max_val[col] < val) {
        _col_to_max_val[col] = val;
      }
      if (_col_to_min_val[col] > val) {
        _col_to_min_val[col] = val;
      }
    } catch (std::invalid_argument& e) {
      throw std::invalid_argument(
          "Could not process column " + std::to_string(col) +
          " as type 'numeric.' Received value: '" + str_value + ".'");
    }
  }

  void processLabel(const std::string& class_name) {
    if (!_class_to_class_id.count(class_name)) {
      if (_class_to_class_id.size() == _n_classes) {
        throw std::invalid_argument(
            "Expected " + std::to_string(_n_classes) +
            " classes but found an additional class: '" + class_name + ".'");
      }
      _class_to_class_id[class_name] = _class_to_class_id.size();
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
  uint32_t _n_classes;
  std::vector<TabularDataType> _column_dtypes;
  std::unordered_map<uint32_t, double> _col_to_max_val;
  std::unordered_map<uint32_t, double> _col_to_min_val;
  std::vector<std::string> _column_names;
  std::unordered_map<std::string, uint32_t> _class_to_class_id;
};

}  // namespace thirdai::dataset
