#pragma once

#include <dataset/src/utils/SafeFileIO.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace thirdai::dataset {

class ProcessorUtils {
 public:
  static std::vector<std::string> aggregateSingleColumnCsvRows(
      const std::string& file_name, uint32_t column_index,
      bool has_header = false, char delimiter = ',');

 private:
  static bool subStringIsQuoted(const std::string& row, size_t start,
                                size_t len);
};

}  // namespace thirdai::dataset