#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/utils/TokenEncoding.h>

namespace thirdai::dataset {

/**
 * Featurizes text data for next word prediction.
 */
class TextGenerationProcessor {
 public:
  TextGenerationProcessor(uint32_t seq_len, uint32_t input_dim,
                          uint32_t output_dim);

  /**
   * Featurizes a list of rows from a text dataset for next word prediction.
   * Expects that each line will be a json string with the text stored under the
   * key "text", this is the format used by hugging face text datasets and the
   * Pile dataset. Returns a list of the input samples and labels for the
   * documents/texts in the input rows.
   *
   * For every N+1 consecutive words the first N words are featurized with
   * pairgrams as the input and the label is the (N+1)th word. The labels are
   * encoded using two hashes of the target word such that a prediction can be
   * recovered by looking at the intersection of the words mapped to the given
   * hash indices.
   */
  std::pair<std::vector<BoltVector>, std::vector<BoltVector>> featurize(
      const std::vector<std::string>& lines) const;

 private:
  /**
   * Helper function to featurize a single line from the text dataset and
   * returns the created input samples and labels.
   */
  std::pair<std::vector<BoltVector>, std::vector<BoltVector>> featurizeText(
      const std::string& line) const;

  /**
   * Helper function to remove all the punctuation from a string and converts
   * any whitespace to ' ' characters so they do not affect the output tokens.
   * The ' ' extra characters will be ignored when splitting the words.
   */
  static std::string removePunctuationAndSpacing(const std::string& str);

  uint32_t _seq_len;
  uint32_t _input_dim;
  uint32_t _output_dim;
};

}  // namespace thirdai::dataset