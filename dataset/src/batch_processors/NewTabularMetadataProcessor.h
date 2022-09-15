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

struct TabularColumnMetadata {
  virtual void update(const std::string& value){};

  virtual ~ColumnMetadata() = default;
};

struct NumericColumnMetadata : public TabularColumnMetadata {
  NumericColumnMetadata(double _min, double _max) : min(_min), max(_max) {}

  NumericColumnMetadata()
      : min(std::numeric_limits<double>::max()),
        max(std::numeric_limits<double>::min()) {}

  void update(const std::string& value) final {
    // TODO(david) handle nan/inf/large number cases

    // we want to process empty values and put them in their own bin later, thus
    // we don't fail here
    if (value.empty()) {
      return;
    }
    try {
      double val = std::stod(str_value);
      if (val > max) max = val;
      if (val < min) min = val;
    } catch (std::invalid_argument& e) {
      throw std::invalid_argument(
          "Could not process column " + std::to_string(col) +
          " as type 'numeric.' Received value: '" + str_value + ".'");
    }
  }

  double min;
  double max;
};

struct CategoricalColumnMetadata : public TabularColumnMetadata {
  CategoricalColumnMetadata() {}
};

struct LabelColumnMetadata : public TabularColumnMetadata {
  explicit LabelColumnMetadata(uint32_t _n_classes) : n_classes(_n_classes) {}

  explicit LabelColumnMetadata(std::vector<std::string> class_names)
      : n_classes(class_names.size()),
        _class_id_to_class(std::move(class_names)) {
    for (uint32_t class_id = 0; class_id < _class_id_to_class.size();
         class_id++) {
      _class_to_class_id[_class_id_to_class[class_id]] = class_id;
    }
  }

  void update(const std::string& value) final {
    uint32_t class_id = _class_id_to_class.size();
    if (class_id == n_classes) {
      throw std::invalid_argument("Expected " + std::to_string(n_classes) +
                                  " classes but found an additional class: '" +
                                  value + ".'");
    }
    _class_to_class_id[value] = class_id;
    _class_id_to_class.push_back(value);
  }

  uint32_t n_classes;
  std::unordered_map<std::string, uint32_t> _class_to_class_id;
  std::vector<std::string> _class_id_to_class;
};

/**
 * @brief This class stores metadata about a tabular csv dataset. This
 * includes column datatype information, min/max information for numeric
 * columns, number of unique values, etc. This metadata is used in the
 * TabularPairgram block to featurize future tabular datasets.
 */
struct TabularMetadata {
  TabularMetadata(std::vector<TabularColumnMetadata> _column_metadata,
                  std::unordered_map<uint32_t, std::string> _col_to_name)
      : column_metadata(std::move(_column_metadata)),
        col_to_name(std::move(_col_to_name)) {}

  std::vector<TabularColumnMetadata> column_metadata;
  std::unordered_map<uint32_t, std::string> col_to_name;
};

/**
 * @brief This class defined a batch processor to collect metadata about a
 * tabular csv dataset. This metadata is in the form of the TabularMetadata
 * object and can be retrieved at the end of processing. Since this class
 * inherets ComputeBatchProcessor no rows are returned, this is simply a means
 * of iterating through a dataset.
 *
 * @param column_types Vector of column types, should be one of label,
 * categorical, or numeric.
 * @param n_classes The expected number of classes in the dataset/
 */
class TabularMetadataProcessor : public ComputeBatchProcessor {
 public:
  explicit TabularMetadataProcessor(
      const std::vector<std::string>& column_types, uint32_t n_classes,
      char delimiter = ',')
      : _delimiter(delimiter) {
    bool found_label_column = false;

    for (const auto& col_type : column_types) {
      if (col_type == "label") {
        if (found_label_column) {
          throw std::invalid_argument(
              "Specified multiple label columns, only one accepted.");
        }
        found_label_column = true;
        _column_metadata.push_back(LabelColumnMetadata(n_classes));
      } else if (col_type == "categorical") {
        _column_metadata.push_back(CategoricalColumnMetadata());
      } else if (col_type == "numeric") {
        _column_metadata.push_back(NumericColumnMetadata());
      } else {
        throw std::invalid_argument(
            "Received datatype '" + col_type +
            "' expected one of 'label', 'categorical', or 'numeric'.");
      }
    }
  }

  bool expectsHeader() const final { return true; }

  void processHeader(const std::string& header) final {
    std::vector<std::string_view> column_names =
        ProcessorUtils::parseCsvRow(header, _delimiter);
    verifyNumColumns(column_names);
    for (uint32_t col = 0; col < column_names.size(); col++) {
      _col_to_name[col] = column_names[col];
    }
  }

  void processRow(const std::string& row) final {
    std::vector<std::string_view> values =
        ProcessorUtils::parseCsvRow(row, _delimiter);
    verifyNumColumns(values);
    for (uint32_t col = 0; col < values.size(); col++) {
      std::string str_value(values[col]);
      _column_metadata[col].update(str_value);
    }
  }

  std::shared_ptr<TabularMetadata> getMetadata() {
    return std::make_shared<TabularMetadata>(_column_metadata, _col_to_name);
  }

 private:
  void verifyNumColumns(const std::vector<std::string_view>& values) {
    if (values.size() != _column_metadata.size()) {
      throw std::invalid_argument("Csv format error. Expected " +
                                  std::to_string(_column_metadata.size()) +
                                  " columns but received " +
                                  std::to_string(values.size()) + " columns.");
    }
  }

  char _delimiter;
  std::vector<TabularColumnMetadata> _column_metadata;
  std::unordered_map<uint32_t, std::string> _col_to_name;
};

}  // namespace thirdai::dataset