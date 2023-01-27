#include "ProcessorUtils.h"
#include <dataset/src/utils/SafeFileIO.h>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace thirdai::dataset {

std::vector<std::string_view> ProcessorUtils::parseCsvRow(
    const std::string& row, char delimiter) {
  std::vector<std::string_view> parsed;
  size_t start = 0;
  size_t end = 0;
  while (end != std::string::npos) {
    end = row.find(delimiter, start);
    size_t len = end == std::string::npos ? row.size() - start : end - start;

    // We wish to trim whitespace on each item to make parsing easier (this
    // also includes new lines). Since we store the parsed csv element as a
    // stringview of the original row, we can trim leading whitespace
    // by incrementing the start index and decrementing the length, and trim
    // trailing whitespace by decrementing the length.
    while (len > 0 && std::isspace(row.at(start))) {
      start++;
      len--;
    }
    while (len > 0 && std::isspace(row.at(start + len - 1))) {
      len--;
    }

    // We also wish to trim quotes (either "" or '') from the string, which
    // we can do in a similar way by incrementing the start and decreasing the
    // length by 2 as long as the beginning and end of the stringview are
    // matching quotes
    while (len > 1 && subStringIsQuoted(row, start, len)) {
      start++;
      len -= 2;
    }

    parsed.push_back(std::string_view(row.data() + start, len));

    start = end + 1;
  }
  return parsed;
}

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
        std::string(parseCsvRow(row, delimiter)[column_index]);
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