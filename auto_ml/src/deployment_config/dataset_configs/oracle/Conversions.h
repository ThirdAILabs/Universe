#pragma once

#include "Aliases.h"
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace thirdai::automl::deployment {

struct ConversionUtils {
  static std::vector<std::string_view> stringInputToVectorOfStringViews(
      const std::string& input_string, char delimiter) {
    return dataset::ProcessorUtils::parseCsvRow(input_string, delimiter);
  }

  static std::vector<std::string_view> mapInputToVectorOfStringViews(
      const std::unordered_map<std::string, std::string>& input_map,
      const ColumnNumberMap& column_number_map) {
    std::vector<std::string_view> string_view_input(
        column_number_map.numCols());
    for (const auto& [col_name, val] : input_map) {
      string_view_input[column_number_map.at(col_name)] =
          std::string_view(val.data(), val.length());
    }
    return string_view_input;
  }

  static std::vector<std::string> mapVectorInputsToVectorOfStrings(
      const std::vector<std::unordered_map<std::string, std::string>>&
          input_maps,
      char delimiter, const ColumnNumberMap& column_number_map) {
    std::vector<std::string> string_batch(input_maps.size());
    for (uint32_t i = 0; i < input_maps.size(); i++) {
      auto vals =
          mapInputToVectorOfStringViews(input_maps[i], column_number_map);
      std::stringstream s;
      std::copy(vals.begin(), vals.end(),
                std::ostream_iterator<std::string_view>(s, &delimiter));
      string_batch[i] = s.str();
    }
    return string_batch;
  }
};

}  // namespace thirdai::automl::deployment