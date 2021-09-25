#pragma once
#include "StringLoader.h"
#include <algorithm>
#include <fstream>
#include <iostream>

/**
 * For September 24th:
 * - Loads strings from files based on queue (manages which file in the queue is
 * being read)
 * - No stemming
 * - Converts non-newline whitespaces to space
 * - Converts uppercase letters to lowercase
 * - Removes every character that is not a letter, a number, space, or sentence
 * ending punctuations
 * - Splits string into a vector of strings, using ending punctuations (.?!) as
 * delimiters
 */
namespace thirdai::utils {
class SentenceLoader : public StringLoader {
 public:
  /**
   * Inherits String Loader.
   */
  SentenceLoader(){};

  bool loadNextString(std::string& str_buf) override;

 private:
  std::ifstream _file;
  std::string _line_buffer;
  size_t _lb_idx = 0;
  size_t _queue_idx = 0;

  static void cleanUpLineBuffer(std::string& line_buffer);

  bool getNextLine(std::string& next_line_buf);
};
}  // namespace thirdai::utils