#pragma once

#include <dataset/src/utils/SafeFileIO.h>
#include <fstream>
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
      parsed.push_back(std::string_view(row.data() + start, len));
      start = end + 1;
    }
    return parsed;
  }

  static std::vector<std::string> aggregateSingleColumnCsvRows(
      const std::string& file_name, uint32_t column_index) {
    std::vector<std::string> aggregated_rows;

    try {
      std::ifstream input_file_stream =
          dataset::SafeFileIO::ifstream(file_name, std::ios::in);

      std::string row, target_column;
      while (std::getline(input_file_stream, row)) {
        target_column = std::string(parseCsvRow(row, ',')[column_index]);
        aggregated_rows.emplace_back(std::move(target_column));
      }
    } catch (const std::ifstream::failure& exception) {
      throw std::invalid_argument("Invalid input file name.");
    }

    return aggregated_rows;
  }
};

}  // namespace thirdai::dataset