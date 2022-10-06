#pragma once

#include <bolt/src/auto_classifiers/sequential_classifier/ConstructorUtilityTypes.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thirdai::automl::deployment {

using DataType = bolt::sequential_classifier::DataType;

class ColumnNumberMap {
 public:
  ColumnNumberMap(const std::string& header, char delimiter) {
    auto header_columns =
        dataset::ProcessorUtils::parseCsvRow(header, delimiter);
    for (uint32_t col_num = 0; col_num < header_columns.size(); col_num++) {
      std::string col_name(header_columns[col_num]);
      _name_to_num[col_name] = col_num;
    }
  }

  ColumnNumberMap() {}

  uint32_t at(const std::string& col_name) const {
    if (_name_to_num.count(col_name) == 0) {
      std::stringstream error_ss;
      error_ss << "Expected a column named '" << col_name
               << "' in header but could not find it.";
      throw std::runtime_error(error_ss.str());
    }
    return _name_to_num.at(col_name);
  }

  size_t size() const { return _name_to_num.size(); }

 private:
  std::unordered_map<std::string, uint32_t> _name_to_num;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_name_to_num);
  }
};

}  // namespace thirdai::automl::deployment