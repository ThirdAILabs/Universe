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

struct TabularColumn {
  explicit TabularColumn(uint32_t _col_num) : col_num(_col_num) {}

  virtual void update(const std::string& str_val) { (void)str_val; };

  virtual std::exception_ptr computeUnigram(const std::string& str_val,
                                            uint32_t& unigram) const = 0;

  virtual ~TabularColumn() = default;

  uint32_t col_num;

 protected:
  // Private constructor for cereal, for use by derived classes
  TabularColumn() {}

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(col_num);
  }
};

using TabularColumnPtr = std::shared_ptr<TabularColumn>;

struct NumericColumn : public TabularColumn {
  NumericColumn(uint32_t _col_num, double _min, double _max,
                uint32_t _num_bins = 10)
      : TabularColumn(_col_num), min(_min), max(_max), num_bins(_num_bins) {}

  NumericColumn(uint32_t _col_num, uint32_t _num_bins = 10)
      : TabularColumn(_col_num),
        min(std::numeric_limits<double>::max()),
        max(std::numeric_limits<double>::min()),
        num_bins(_num_bins) {}

  void update(const std::string& str_val) final {
    // TODO(david) handle nan/inf/large number cases

    // we want to process empty values and put them in their own bin later, thus
    // we don't fail here
    if (str_val.empty()) {
      return;
    }
    try {
      double num = std::stod(str_val);
      if (num > max) {
        max = num;
      }
      if (num < min) {
        min = num;
      }
    } catch (std::invalid_argument& e) {
      throw std::invalid_argument(
          "Could not process column " + std::to_string(col_num) +
          " as type 'numeric.' Received value: '" + str_val + ".'");
    }
  }

  std::exception_ptr computeUnigram(const std::string& str_val,
                                    uint32_t& unigram) const final {
    uint32_t bin;
    if (auto err = getBin(str_val, bin)) {
      return err;
    }
    // set the first half of a uint64_t to the bin and the second half to the
    // column number. the column number acts as salt to make bins unique across
    // columns. we then hash this unique bin
    uint64_t uniqueBin =
        static_cast<uint64_t>(bin) << 32 | static_cast<uint64_t>(col_num);
    const char* val_to_hash = reinterpret_cast<const char*>(&uniqueBin);
    unigram = TextEncodingUtils::computeUnigram(val_to_hash, /* len = */ 8);
    return nullptr;
  }

  double min;
  double max;
  // an extra bin is reserved for empty values
  uint32_t num_bins;

 private:
  std::exception_ptr getBin(const std::string& str_val, uint32_t& bin) const {
    // map empty values to their own bin (the last bin)
    if (str_val.empty()) {
      bin = num_bins;
      return nullptr;
    }
    double value;
    try {
      value = std::stod(str_val);
    } catch (std::invalid_argument& e) {
      return std::make_exception_ptr(std::invalid_argument(
          "Could not process column " + std::to_string(col_num) +
          " as type 'numeric.' Received value: '" + str_val + ".'"));
    }
    double binsize = max - min / num_bins;
    if (binsize == 0) {
      bin = 0;
    } else {
      bin = static_cast<uint32_t>(std::round((value - min) / binsize));
    }
    return nullptr;
  }

  // Private constructor for cereal.
  NumericColumn() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TabularColumn>(this), min, max, num_bins);
  }
};

struct CategoricalColumn : public TabularColumn {
  explicit CategoricalColumn(uint32_t _col_num) : TabularColumn(_col_num) {}

  std::exception_ptr computeUnigram(const std::string& str_val,
                                    uint32_t& unigram) const final {
    const char* char_salt = reinterpret_cast<const char*>(&col_num);
    std::string str_salt(char_salt, 4);
    std::cout << "PRINTING STR SALT" << str_salt << std::endl;
    std::string unique_category = str_val + str_salt;
    unigram = TextEncodingUtils::computeUnigram(unique_category.data(),
                                                unique_category.size());
    return nullptr;
  }

 private:
  // Private constructor for cereal.
  CategoricalColumn() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TabularColumn>(this));
  }
};

struct LabelColumn : public TabularColumn {
  LabelColumn(uint32_t _col_num, uint32_t _n_classes)
      : TabularColumn(_col_num), n_classes(_n_classes) {}

  LabelColumn(uint32_t _col_num, std::vector<std::string> class_names)
      : TabularColumn(_col_num),
        n_classes(class_names.size()),
        class_id_to_class(std::move(class_names)) {
    for (uint32_t class_id = 0; class_id < class_id_to_class.size();
         class_id++) {
      class_to_class_id[class_id_to_class[class_id]] = class_id;
    }
  }

  void update(const std::string& str_val) final {
    uint32_t class_id = class_id_to_class.size();
    if (class_id == n_classes) {
      throw std::invalid_argument("Expected " + std::to_string(n_classes) +
                                  " classes but found an additional class: '" +
                                  str_val + ".'");
    }
    class_to_class_id[str_val] = class_id;
    class_id_to_class.push_back(str_val);
  }

  std::exception_ptr computeUnigram(const std::string& str_val,
                                    uint32_t& unigram) const final {
    (void)str_val;
    (void)unigram;
    return std::make_exception_ptr(std::invalid_argument(
        "Shouldn't compute unigram of LabelColumn, not supported. "));
  }

  uint32_t n_classes;
  std::unordered_map<std::string, uint32_t> class_to_class_id;
  std::vector<std::string> class_id_to_class;

 private:
  // Private constructor for cereal.
  LabelColumn() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TabularColumn>(this), n_classes,
            class_to_class_id, class_id_to_class);
  }
};

/**
 * @brief This class stores metadata about a tabular csv dataset. This
 * includes column datatype information, min/max information for numeric
 * columns, etc. This metadata is used in the TabularClassifier to featurize
 * future tabular datasets.
 */
struct TabularMetadata {
  TabularMetadata(const std::vector<TabularColumnPtr>& _column_metadata,
                  std::unordered_map<uint32_t, std::string> _col_to_name)
      : col_to_name(std::move(_col_to_name)) {
    for (const auto& tabular_column : _column_metadata) {
      if (auto casted_label_column =
              std::dynamic_pointer_cast<LabelColumn>(tabular_column)) {
        label_column = casted_label_column;
      } else {
        non_label_columns.push_back(tabular_column);
      }
    }
  }

  std::shared_ptr<LabelColumn> label_column;
  std::vector<TabularColumnPtr> non_label_columns;
  std::unordered_map<uint32_t, std::string> col_to_name;

 private:
  // Private constructor for cereal
  TabularMetadata() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(label_column, non_label_columns, col_to_name);
  }
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

    for (uint32_t col = 0; col < column_types.size(); col++) {
      const auto& col_type = column_types[col];
      if (col_type == "label") {
        if (found_label_column) {
          throw std::invalid_argument(
              "Specified multiple label columns, only one accepted.");
        }
        found_label_column = true;
        _column_metadata.push_back(
            std::make_shared<LabelColumn>(col, n_classes));
      } else if (col_type == "categorical") {
        _column_metadata.push_back(std::make_shared<CategoricalColumn>(col));
      } else if (col_type == "numeric") {
        _column_metadata.push_back(std::make_shared<NumericColumn>(col));
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
      _column_metadata[col]->update(str_value);
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
  std::vector<TabularColumnPtr> _column_metadata;
  std::unordered_map<uint32_t, std::string> _col_to_name;
};

}  // namespace thirdai::dataset