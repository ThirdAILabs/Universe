#include "ReservedColumns.h"
#include <regex>
#include <stdexcept>

namespace thirdai::automl {

bool isReservedColumnName(const std::string& name) {
  std::regex re("__.*__");

  return std::regex_match(name, re);
}

void checkNoReservedColumnNames(const ColumnDataTypes& data_types) {
  for (const auto& [name, _] : data_types) {
    if (isReservedColumnName(name)) {
      throw std::invalid_argument("Column name '" + name +
                                  "' is a reserved column name. Input columns "
                                  "cannot start and end with '__'");
    }
  }
}

inline std::string outputColumnName(const std::string& name,
                                    const std::string& type) {
  return "__" + name + "_" + type + "__";
}

std::string textOutputColumn(const std::string& input_column_name) {
  return outputColumnName(input_column_name, "tokenized");
}

std::string categoricalOutputColumn(const std::string& input_column_name) {
  return outputColumnName(input_column_name, "categorical");
}

std::string binningOutputColumn(const std::string& input_column_name) {
  return outputColumnName(input_column_name, "binned");
}

std::string sequenceOutputColumn(const std::string& input_column_name) {
  return outputColumnName(input_column_name, "sequence");
}

std::string dateOutputColumn(const std::string& input_column_name) {
  return outputColumnName(input_column_name, "date");
}

std::string temporalItemIdsOutput(const std::string& input_column_name) {
  return outputColumnName(input_column_name, "item_ids");
}

std::string temporalNumericalValueOutput(const std::string& input_column_name) {
  return outputColumnName(input_column_name, "numerical_value");
}

std::string temporalTrackingOutput(uint32_t temporal_id) {
  return "__categorical_temporal_" + std::to_string(temporal_id) + "__";
}

}  // namespace thirdai::automl