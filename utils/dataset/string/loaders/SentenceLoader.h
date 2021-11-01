#pragma once
#include "StringLoader.h"
#include <algorithm>
#include <fstream>
#include <iostream>

/**
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
namespace thirdai::utils::dataset {
class SentenceLoader : public StringLoader {
 public:
  /**
   * Inherits String Loader.
   * Sentence loader loads batches of sentences from a file.
   * It assumes each sentence has no label, so loaded_labels
   * only contains empty vector.s
   */
  SentenceLoader(){};

  uint32_t loadStringsAndLabels(
      std::ifstream& file, uint32_t target_batch_size,
      std::vector<std::string>& loaded_strings,
      std::vector<std::vector<uint32_t>>& loaded_labels) override;

  virtual ~SentenceLoader() {}

 private:
  std::string _line_buffer;
  size_t _lb_idx = 0;

  /**
   * Helper function that loads the next sentence from file into str_buf.
   */
  bool loadNextSentence(std::ifstream& file, std::string& str_buf);

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
};
}  // namespace thirdai::utils::dataset