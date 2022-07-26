#pragma once

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
};

}  // namespace thirdai::dataset