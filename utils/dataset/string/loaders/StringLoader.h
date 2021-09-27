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
   * Loads the next string (sentence, paragraph, or the whole file) from
   * the first file in _filename_queue that has not been exhausted
   * and into 'str_buf', overwriting the string.
   * The string can only contain lower case characters, numbers, and space.
   * All punctuation marks must be stripped off.
   * Returns whether the next string is loaded successfully.
   */
  virtual bool loadNextString(std::string& str_buf) = 0;

  /**
   * Adds a file to a queue of files to be read from.
   */
  void addFileToQueue(std::string& filename) {
    _filename_queue.push_back(filename);
  };

 protected:
  /**
   * A queue of filenames for strings to be loaded from.
   */
  std::vector<std::string> _filename_queue;
};
}  // namespace thirdai::utils