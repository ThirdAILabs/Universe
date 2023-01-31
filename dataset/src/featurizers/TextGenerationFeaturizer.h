#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Featurizer.h>
#include <limits>

namespace thirdai::dataset {

/**
 * Featurizes text data for next word prediction.
 */
class TextGenerationFeaturizer final : public Featurizer {
 public:
  TextGenerationFeaturizer(uint32_t sequence_len, uint32_t vocab_size);

  /**
   * Featurizes a list of rows from a text dataset for next word prediction.
   * Expects that each line will be a list of space separate bert tokens
   * (integer ids) for the text of a given document/text. Returns a list of
   * the input samples and labels for the documents/texts in the input rows.
   *
   * For every N+1 consecutive words the first N words are featurized with
   * pairgrams as the input and the label is the (N+1)th word.
   *
   * For example the tokens [1, 2, 3, 4, 5, 6] with sequence_len=4 will give the
   * following samples:
   *
   * input=pairgrams(1, 2, 3, 4), label=5
   * input=pairgrams(2, 3, 4, 5), label=6
   */
  std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& lines) final;

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

  size_t getNumDatasets() final { return 2; }

  std::vector<uint32_t> getDimensions() final {
    return {std::numeric_limits<uint32_t>::max(), _vocab_size};
  }

 private:
  /**
   * Helper function to featurize a single line from the text dataset and
   * returns the created input samples and labels.
   */
  std::pair<std::vector<BoltVector>, std::vector<BoltVector>> featurizeText(
      const std::string& line) const;

  static std::vector<uint32_t> parseTokens(const std::string& line);

  uint32_t _sequence_len;
  uint32_t _vocab_size;
};

}  // namespace thirdai::dataset