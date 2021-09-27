#pragma once
#include <fstream>
#include <iostream>
#include <string>

namespace thirdai::utils {
/**
 * Interface to open text files and extract strings of a certain type,
 * such as sentences, paragraphs, or the whole file.
 */
class StringLoader {
 public:
  /**
   * Loads the next string (sentence, paragraph, or the whole file) from a file
   * and into 'str_buf', overwriting the string.
   * The string can only contain lower case characters, numbers, and space.
   * All punctuation marks must be stripped off.
   * Returns whether the next string is loaded successfully.
   */
  virtual bool loadNextString(std::string& str_buf) = 0;
  virtual void updateFile(std::string filename) = 0;
};
}  // namespace thirdai::utils