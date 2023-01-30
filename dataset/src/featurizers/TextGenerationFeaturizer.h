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
  TextGenerationFeaturizer(uint32_t seq_len, uint32_t output_dim);

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
  std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& lines) final;

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

  size_t getNumDatasets() final { return 2; }

  std::vector<uint32_t> getDimensions() final {
    return {std::numeric_limits<uint32_t>::max(), _output_dim};
  }

 private:
  /**
   * Helper function to featurize a single line from the text dataset and
   * returns the created input samples and labels.
   */
  std::pair<std::vector<BoltVector>, std::vector<BoltVector>> featurizeText(
      const std::string& line) const;

  static std::vector<uint32_t> parseTokens(const std::string& line);

  uint32_t _seq_len;
  uint32_t _output_dim;
};

}  // namespace thirdai::dataset