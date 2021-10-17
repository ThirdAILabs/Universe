#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace thirdai::utils {
/**
 * Interface to open text files and extract strings of a certain type,
 * such as sentences, paragraphs, or the whole file.
 */
class StringLoader {
 public:
  /**
   * Loads up to target_batch_size strings from file into 'loaded_strings',
   * overwriting its contents
   * The string can only contain lower case characters, numbers, and space.
   * All punctuation marks must be stripped off.
   */
  virtual void loadStrings(std::ifstream& file, uint32_t target_batch_size,
                           std::vector<std::string>& loaded_strings) = 0;
};
}  // namespace thirdai::utils