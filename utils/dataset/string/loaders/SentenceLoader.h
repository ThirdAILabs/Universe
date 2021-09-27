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

  /**
   * Cleans up the line buffer.
   * - Convert all whitespaces into space.
   * - Convert all uppercase characters into lowercase.
   * - Convert all end-of-sentence punctuation marks to periods.
   * - Remove any character that is not a letter, a number, or space.
   * - Remove duplicate spaces and periods.
   * - Remove periods and spaces before the first letter or number.
   * - Remove trailing periods and spaces.
   */
  static void cleanUpLineBuffer(std::string& line_buffer);

  /**
   * Retrieves the next line from the queue of files;
   * Keep doing the following until a line is found:
   *  If the current file is not exhausted, retrieve the next line in the file.
   *  Otherwise, close the current file, open the next file and read a line from
   * that file. If no line is found, return false.
   */
  bool getNextLine(std::string& next_line_buf);
};
}  // namespace thirdai::utils