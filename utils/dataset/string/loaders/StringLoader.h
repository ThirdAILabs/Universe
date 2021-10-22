#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace thirdai::utils::dataset {
/**
 * Interface to open text files and extract strings of a certain type,
 * such as sentences, paragraphs, or the whole file.
 */
class StringLoader {
 public:
  /**
   * Loads up to target_batch_size strings from file into 'loaded_strings',
   * overwriting its contents, and up to target_batch_size label vectors
   * from file into 'loaded_labels', overwriting its contents.
   */
  virtual void loadStringsAndLabels(
      std::ifstream& file, uint32_t target_batch_size,
      std::vector<std::string>& loaded_strings,
      std::vector<std::vector<uint32_t>>& loaded_labels) = 0;
};
}  // namespace thirdai::utils::dataset