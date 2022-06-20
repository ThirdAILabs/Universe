#pragma once

#include <fstream>

namespace thirdai::dataset {

class FileVerifier {
 public:
  static void verifyFile(const std::ifstream& file,
                         const std::string& filename) {
    if (file.bad() || file.fail() || !file.good() || !file.is_open()) {
      throw std::runtime_error("Unable to open file '" + filename + "'");
    }
  }
};

}  // namespace thirdai::dataset
