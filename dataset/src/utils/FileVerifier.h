#pragma once

#include <fstream>

namespace thirdai::dataset {

class FileUtils {
 public:
  static std::ifstream safeInputFile(const std::string& filename) {
    std::ifstream file(filename);
    if (file.bad() || file.fail() || !file.good() || !file.is_open()) {
      throw std::runtime_error("Unable to open file '" + filename + "'");
    }
    return file;
  }

  static std::ofstream safeOutputFile(const std::string& filename) {
    std::ofstream file(filename);
    if (file.bad() || file.fail() || !file.good() || !file.is_open()) {
      throw std::runtime_error("Unable to open file '" + filename + "'");
    }
    return file;
  }

  static std::fstream safeFile(const std::string& filename) {
    std::fstream file(filename);
    if (file.bad() || file.fail() || !file.good() || !file.is_open()) {
      throw std::runtime_error("Unable to open file '" + filename + "'");
    }
    return file;
  }
};

}  // namespace thirdai::dataset
