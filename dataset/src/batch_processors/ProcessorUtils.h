#pragma once

#include <dataset/src/utils/SafeFileIO.h>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace thirdai::dataset {

class ProcessorUtils {
 public:
  static std::vector<std::string_view> parseCsvRow(const std::string& row,
                                                   char delimiter) {
    std::vector<std::string_view> parsed;
    size_t start = 0;
    size_t end = 0;
    while (end != std::string::npos) {
      end = row.find(delimiter, start);
      size_t len = end == std::string::npos ? row.size() - start : end - start;

      while (len > 0 && std::isspace(row.at(start))) {
        start++;
        len--;
      }
      while (len > 0 && std::isspace(row.at(start + len - 1))) {
        len--;
      }
      while (len > 1 && row.at(start) == '"' &&
             row.at(start + len - 1) == '"') {
        start++;
        len -= 2;
      }

      parsed.push_back(std::string_view(row.data() + start, len));

      start = end + 1;
    }
    return parsed;
  }

  static std::vector<std::string> aggregateSingleColumnCsvRows(
      const std::string& file_name, uint32_t column_index,
      bool has_header = false) {
    std::vector<std::string> aggregated_rows;

    std::ifstream input_file_stream =
        dataset::SafeFileIO::ifstream(file_name, std::ios::in);

    std::string row;
    if (has_header) {
      std::getline(input_file_stream, row);
    }
    while (std::getline(input_file_stream, row)) {
      std::string target_column =
          std::string(parseCsvRow(row, ',')[column_index]);
      aggregated_rows.emplace_back(std::move(target_column));
    }

    return aggregated_rows;
  }
};

}  // namespace thirdai::dataset