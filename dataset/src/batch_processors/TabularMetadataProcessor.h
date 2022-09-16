#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
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

class TabularMetadataProcessor : public ComputeBatchProcessor {
 public:
  explicit TabularMetadataProcessor(std::vector<std::string> column_types,
                                    uint32_t n_classes)
      : _delimiter(','),
        _label_vocab(ThreadSafeVocabulary::make(/* vocab_size = */ n_classes)) {
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
          _label_vocab->getUid(str_value);
          break;
        }
      }
    }
  }

  std::shared_ptr<TabularMetadata> getMetadata() {
    return std::make_shared<TabularMetadata>(_column_dtypes, _col_to_max_val,
                                             _col_to_min_val, _label_vocab);
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

  void verifyNumColumns(const std::vector<std::string_view>& values) {
    if (values.size() != _column_dtypes.size()) {
      throw std::invalid_argument("Csv format error. Expected " +
                                  std::to_string(_column_dtypes.size()) +
                                  " columns but received " +
                                  std::to_string(values.size()) + " columns.");
    }
  }

  char _delimiter;
  std::vector<TabularDataType> _column_dtypes;
  std::unordered_map<uint32_t, double> _col_to_max_val;
  std::unordered_map<uint32_t, double> _col_to_min_val;
  std::vector<std::string> _column_names;
  ThreadSafeVocabularyPtr _label_vocab;
};

}  // namespace thirdai::dataset
