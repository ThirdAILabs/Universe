#include "ProcessorUtils.h"
#include <dataset/src/utils/CsvParser.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace thirdai::dataset {

std::vector<std::string> ProcessorUtils::aggregateSingleColumnCsvRows(
    const std::string& file_name, uint32_t column_index, bool has_header,
    char delimiter) {
  std::vector<std::string> aggregated_rows;

  std::ifstream input_file_stream =
      dataset::SafeFileIO::ifstream(file_name, std::ios::in);

  std::string row;
  if (has_header) {
    std::getline(input_file_stream, row);
  }
  while (std::getline(input_file_stream, row)) {
    std::string target_column =
        std::string(parsers::CSV::parseLine(row, delimiter)[column_index]);
    aggregated_rows.emplace_back(std::move(target_column));
  }

  return aggregated_rows;
}

bool ProcessorUtils::subStringIsQuoted(const std::string& row, size_t start,
                                       size_t len) {
  return (row.at(start) == '"' && row.at(start + len - 1) == '"') ||
         (row.at(start) == '\'' && row.at(start + len - 1) == '\'');
}

}  // namespace thirdai::dataset