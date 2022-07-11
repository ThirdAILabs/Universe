#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <dataset/src/bolt_datasets/BatchProcessor.h>
#include <cmath>
#include <limits>

namespace thirdai::dataset {

enum class TabularDataType {
  Numeric,
  Categorical,
  Label
};  // TODO(david) add datetime/text support

class TabularMetadata {
  friend class TabularMetadataProcessor;

 public:
  TabularMetadata() {}

  uint32_t numColumns() const { return _column_dtypes.size(); }

  TabularDataType getType(uint32_t col) const { return _column_dtypes[col]; }

  uint32_t numClasses() const { return _class_id_to_class.size(); }

  uint32_t getLabelCol() const { return _label_col_index; }

  std::vector<std::string> getClassIdToNames() { return _class_id_to_class; }

  std::unordered_map<std::string, uint32_t> getClassToIdMap() {
    return _class_to_class_id;
  }

  std::string getColBin(uint32_t col, const std::string& str_val,
                        std::string& block_exception_message) {
    // map empty values to their own bin
    if (str_val.empty()) {
      return std::to_string(_num_non_empty_bins);
    }
    try {
      double value = std::stod(str_val);
      double binsize = getColBinsize(col);
      if (binsize == 0) {
        return "0";
      }
      return std::to_string(static_cast<uint32_t>(
          std::round((value - getColMin(col)) / getColBinsize(col))));
    } catch (std::invalid_argument& e) {
      block_exception_message =
          "Could not process column " + std::to_string(col) +
          " as type 'numeric.' Received value: '" + str_val + ".'";

      // Since we have set the block exception message above, the program will
      // fail once all threads finish. Since we can't throw an exception within
      // a pragma thread, we just have to keep the program running until then.
      // Thus we return some arbitrary value to do that.
      return "0";
    }
  }

  /**
   * To compute pairgrams of categories across columns we need all categories to
   * be unique. This method returns "salt" to make each category unique
   * dependent on its column. By adding the column number padded to a fixed
   * length, we ensure this characteristic.
   */
  std::string getColSalt(uint32_t col) {
    std::string col_str = std::to_string(col);
    if (col_str.size() < _max_salt_len) {
      col_str.insert(0, _max_salt_len - col_str.size(), '0');
    }
    return col_str;
  }

  std::string getClassName(uint32_t class_id) {
    return _class_id_to_class.at(class_id);
  }

 private:
  double getColMin(uint32_t col) { return _col_to_min_val[col]; }

  double getColBinsize(uint32_t col) {
    return (_col_to_max_val[col] - _col_to_min_val[col]) / _num_non_empty_bins;
  }

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_num_non_empty_bins, _label_col_index, _column_dtypes,
            _col_to_max_val, _col_to_min_val, _class_to_class_id,
            _class_id_to_class);
  }

  // one additional bin is reserved for empty values
  uint32_t _num_non_empty_bins = 10;
  uint32_t _label_col_index;
  uint32_t _max_salt_len;
  std::vector<TabularDataType> _column_dtypes;
  std::unordered_map<uint32_t, double> _col_to_max_val;
  std::unordered_map<uint32_t, double> _col_to_min_val;
  std::unordered_map<std::string, uint32_t> _class_to_class_id;
  std::vector<std::string> _class_id_to_class;
};

class TabularMetadataProcessor : public ComputeBatchProcessor {
 public:
  explicit TabularMetadataProcessor(std::vector<std::string> column_datatypes,
                                    uint32_t n_classes)
      : _delimiter(','),
        _n_classes(n_classes),
        _metadata(std::make_shared<TabularMetadata>()) {
    bool found_label_column = false;
    for (uint32_t col_idx = 0; col_idx < column_datatypes.size(); col_idx++) {
      if (column_datatypes[col_idx] == "label" && found_label_column) {
        throw std::invalid_argument(
            "Found multiple 'label' columns in dataset.");
      }
      if (column_datatypes[col_idx] == "label") {
        found_label_column = true;
        _metadata->_label_col_index = col_idx;
        _metadata->_column_dtypes.push_back(TabularDataType::Label);
      } else if (column_datatypes[col_idx] == "categorical") {
        _metadata->_column_dtypes.push_back(TabularDataType::Categorical);
      } else if (column_datatypes[col_idx] == "numeric") {
        _metadata->_column_dtypes.push_back(TabularDataType::Numeric);
        _metadata->_col_to_max_val[col_idx] =
            std::numeric_limits<double>::min();
        _metadata->_col_to_min_val[col_idx] =
            std::numeric_limits<double>::max();
      } else {
        throw std::invalid_argument(
            "Received datatype '" + column_datatypes[col_idx] +
            "' expected one of 'label', 'categorical', or 'numeric'.");
      }
    }
    if (!found_label_column) {
      throw std::invalid_argument("Dataset does not contain a 'label' column.");
    }
    _metadata->_max_salt_len = std::to_string(column_datatypes.size()).size();
  }

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

  void processRow(const std::string& row) final {
    std::vector<std::string_view> values = parseCsvRow(row, _delimiter);
    if (values.size() != _metadata->_column_dtypes.size()) {
      throw std::invalid_argument(
          "Csv format error. Expected " +
          std::to_string(_metadata->_column_dtypes.size()) +
          " columns but received " + std::to_string(values.size()) +
          " columns.");
    }
    for (uint32_t col = 0; col < _metadata->_column_dtypes.size(); col++) {
      std::string str_value(values[col]);
      switch (_metadata->_column_dtypes[col]) {
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
  void processNumeric(std::string str_value, uint32_t col) {
    // TODO(david) handle nan/inf/large number cases

    // we want to process empty values and put them in their own bin later, thus
    // we don't fail here
    if (str_value == "") {
      return;
    }
    try {
      double val = std::stod(str_value);
      if (_metadata->_col_to_max_val[col] < val) {
        _metadata->_col_to_max_val[col] = val;
      }
      if (_metadata->_col_to_min_val[col] > val) {
        _metadata->_col_to_min_val[col] = val;
      }
    } catch (std::invalid_argument& e) {
      throw std::invalid_argument(
          "Could not process column " + std::to_string(col) +
          " as type 'numeric.' Received value: '" + str_value + ".'");
    }
  }

  void processLabel(std::string str_value) {
    if (!_metadata->_class_to_class_id.count(str_value)) {
      uint32_t label = _metadata->_class_id_to_class.size();
      if (_metadata->numClasses() == _n_classes) {
        throw std::invalid_argument(
            "Expected " + std::to_string(_n_classes) +
            " classes but found an additional class: '" + str_value + ".'");
      }
      _metadata->_class_to_class_id[str_value] = label;
      _metadata->_class_id_to_class.push_back(str_value);
    }
  }

  char _delimiter;
  uint32_t _n_classes;
  std::shared_ptr<TabularMetadata> _metadata;
};

}  // namespace thirdai::dataset
