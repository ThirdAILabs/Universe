#pragma once

#include <fstream>

namespace thirdai::dataset {

class SafeFileMaker {
 public:
  static std::ifstream ifstream(
      const std::string& filename,
      std::ios_base::openmode mode = std::ios_base::in) {
    std::ifstream file(filename, mode);
    if (file.bad() || file.fail() || !file.good() || !file.is_open()) {
      throw std::runtime_error("Unable to open input file '" + filename + "'");
    }
    return file;
  }

  static std::ofstream ofstream(
      const std::string& filename,
      std::ios_base::openmode mode = std::ios_base::out) {
    std::ofstream file(filename, mode);
    if (file.bad() || file.fail() || !file.good() || !file.is_open()) {
      throw std::runtime_error("Unable to open output file '" + filename + "'");
    }
    return file;
  }

  static std::fstream fstream(
      const std::string& filename,
      std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out) {
    std::fstream file(filename, mode);
    if (file.bad() || file.fail() || !file.good() || !file.is_open()) {
      throw std::runtime_error("Unable to open file '" + filename + "'");
    }
    return file;
  }
};

}  // namespace thirdai::dataset
