#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {

class ColumnNumberMap {
 public:
  ColumnNumberMap(const std::string& header, char delimiter) : _n_cols(0) {
    auto header_columns =
        dataset::ProcessorUtils::parseCsvRow(header, delimiter);
    _n_cols = header_columns.size();
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

  bool equals(const ColumnNumberMap& other) {
    return other._name_to_num == _name_to_num;
  }

  size_t size() const { return _name_to_num.size(); }

  size_t numCols() const { return _n_cols; }

  std::vector<std::string> getColumnNumToColNameMap() const {
    std::vector<std::string> col_num_to_col_name(numCols());
    for (const auto& [name, num] : _name_to_num) {
      col_num_to_col_name[num] = name;
    }
    return col_num_to_col_name;
  }

 private:
  std::unordered_map<std::string, uint32_t> _name_to_num;
  uint32_t _n_cols;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_name_to_num, _n_cols);
  }
};

using ColumnNumberMapPtr = std::shared_ptr<ColumnNumberMap>;

}  // namespace thirdai::dataset