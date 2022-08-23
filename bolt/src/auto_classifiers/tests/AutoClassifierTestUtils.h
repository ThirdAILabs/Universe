#pragma once

#include <dataset/src/utils/SafeFileIO.h>
#include <fstream>
#include <string>
#include <vector>

namespace thirdai::bolt {

class AutoClassifierTestUtils {
 public:
  static float computePredictFileAccuracy(
      const std::string& pred_filename,
      const std::vector<std::string>& true_labels) {
    std::ifstream file = dataset::SafeFileIO::ifstream(pred_filename);

    float correct_count = 0;
    uint32_t label_idx = 0;
    std::string line;
    while (std::getline(file, line)) {
      if (label_idx == true_labels.size()) {
        throw std::invalid_argument(
            "Received incorrect number of labels. Expected " +
            std::to_string(true_labels.size()) + " but received more.");
      }

      if (line == true_labels[label_idx]) {
        correct_count++;
      }
      label_idx++;
    }
    return correct_count / true_labels.size();
  }

  static void setTempFileContents(const std::string& filename,
                                  std::vector<std::string>& lines) {
    std::ofstream file = dataset::SafeFileIO::ofstream(filename);
    for (const auto& line : lines) {
      file << line << "\n";
    }
    file.close();
  }
};

}  // namespace thirdai::bolt