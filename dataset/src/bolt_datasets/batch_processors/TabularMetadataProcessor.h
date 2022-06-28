#pragma once

#include <dataset/src/bolt_datasets/BatchProcessor.h>

namespace thirdai::dataset {

enum class TabularDataType {
  Numeric,
  Categorical,
  Label
};  // TODO(david) add datetime/text support

// TODO(david) verify each column is valid?
class TabularMetadata {
  friend class TabularMetadataProcessor;

 public:
  TabularMetadata() {}

  uint32_t numColumns() const { return _column_dtypes.size(); }

  TabularDataType getType(uint32_t col) { return _column_dtypes[col]; }

  uint32_t numClasses() const { return _class_id_to_class.size(); }

  std::vector<std::string_view> getColumnNames() { return _column_names; }

  std::string getColBin(uint32_t col, float value) {
    return std::to_string(value - getColMin(col) / getColBinsize(col));
  }

  // TODO(david) check if invalid id
  std::string getClassName(uint32_t class_id) {
    return _class_id_to_class[class_id];
  }
  // TODO(david) check if invalid/new label?? or could do this in parsing
  uint32_t getClassId(const std::vector<std::string_view>& input_row) {
    return _class_to_class_id[input_row[_label_col_index]];
  }

 private:
  float getColMin(uint32_t col) { return _col_to_min_val[col]; }

  float getColBinsize(uint32_t col) {
    return (_col_to_max_val[col] - _col_to_min_val[col]) / _num_bins;
  }

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_num_bins, _label_col_index, _column_names, _column_dtypes,
            _col_to_max_val, _col_to_min_val, _class_to_class_id,
            _class_id_to_class);
  }

  uint32_t _num_bins = 10;
  uint32_t _label_col_index;
  std::vector<std::string_view> _column_names;
  std::vector<TabularDataType> _column_dtypes;
  std::unordered_map<uint32_t, float> _col_to_max_val;
  std::unordered_map<uint32_t, float> _col_to_min_val;
  std::unordered_map<std::string_view, uint32_t> _class_to_class_id;
  std::vector<std::string> _class_id_to_class;
};

class TabularMetadataProcessor : public ComputeBatchProcessor {
 public:
  explicit TabularMetadataProcessor(std::vector<std::string> column_datatypes)
      : _delimiter(','), _metadata(std::make_shared<TabularMetadata>()) {
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
      } else {
        throw std::invalid_argument(
            "Received datatype '" + column_datatypes[col_idx] +
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
        parseCsvRow(header, _delimiter);
    if (column_names.size() != _metadata->_column_dtypes.size()) {
      throw std::invalid_argument(
          "Mismatching number of columns in csv. Based on the datatypes "
          "specified, expected " +
          std::to_string(_metadata->_column_dtypes.size()) +
          " columns but received " + std::to_string(column_names.size()) +
          " columns.");
    }
    _metadata->_column_names = column_names;
  }

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
      switch (_metadata->_column_dtypes[col]) {
        case TabularDataType::Numeric: {
          // TODO(david) if empty, nan, or inf ignore
          // otherwise, report error
          float val = std::stof(std::string(values[col]));
          if (_metadata->_col_to_max_val[val] < val) {
            _metadata->_col_to_max_val[val] = val;
          }
          if (_metadata->_col_to_min_val[val] > val) {
            _metadata->_col_to_min_val[val] = val;
          }
          break;
        }
        case TabularDataType::Categorical:
          break;
        case TabularDataType::Label: {
          if (!_metadata->_class_to_class_id.count(values[col])) {
            uint32_t label = _metadata->_class_id_to_class.size();
            _metadata->_class_to_class_id[values[col]] = label;
            _metadata->_class_id_to_class.push_back(std::string(values[col]));
          }
          break;
        }
      }
    }
  }

  std::shared_ptr<TabularMetadata> getMetadata() { return _metadata; }

 private:
  char _delimiter;
  std::shared_ptr<TabularMetadata> _metadata;
};

}  // namespace thirdai::dataset
