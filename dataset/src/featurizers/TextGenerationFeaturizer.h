#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Featurizer.h>
#include <limits>
#include <stdexcept>

namespace thirdai::dataset {

/**
 * Featurizes a list of rows from a text dataset for next word prediction.
 * Expects that each line will be a list of space separate bert tokens
 * (integer ids) for the text of a given document/text. Returns a list of
 * the input samples and labels for the documents/texts in the input rows.
 *
 * The input samples are composed of 3 parts. The long range context (lrc), the
 * intermediate range context (irc), and the short range context (src). The long
 * range context will be featurized using unigrams, the intermediate range
 * context will be featurized using pairgrams, and the short range context will
 * be featurized with unigrams and padded to a constistent length, this is so
 * these token embeddings can be concatenated in the text generation model.
 *
 *
 * For example the tokens [1, 2, 3, 4, 5, 6] with src_len=4, irc_len=3,
 * src_len=2 will be featurized as follows:
 *
 * +--------------+----------------------+-------------+-------+
 * | LRC Context  | IRC Context          | SRC Context | Label |
 * +--------------+----------------------+-------------+-------+
 * | [1]          | pairgrams([1])       | [0, 1]      | 2     |
 * | [1, 2]       | pairgrams([1, 2])    | [1, 2]      | 3     |
 * | [1, 2, 3]    | pairgrams([1, 2, 3]) | [2, 3]      | 4     |
 * | [1, 2, 3, 4] | pairgrams([2, 3, 4]) | [3, 4]      | 5     |
 * | [2, 3, 4, 5] | pairgrams([3, 4, 5]) | [4, 5]      | 6     |
 * +--------------+----------------------+-------------+-------+
 *
 * Note: note that the first token is assumed to be [CLS], and the token 0
 * denotes [PAD].
 */
class TextGenerationFeaturizer;
using TextGenerationFeaturizerPtr = std::shared_ptr<TextGenerationFeaturizer>;

class TextGenerationFeaturizer final : public Featurizer {
 public:
  TextGenerationFeaturizer(uint32_t lrc_len, uint32_t irc_len, uint32_t src_len)
      : _lrc_len(lrc_len), _irc_len(irc_len), _src_len(src_len) {}

  std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& lines) final;

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

  size_t getNumDatasets() final { return 4; }

  std::vector<uint32_t> getDimensions() final {
    throw std::runtime_error(
        "getDimensions is not supported for TextGenerationFeaturizer.");
  }

  std::vector<BoltVector> featurizeInferenceSample(
      const std::vector<uint32_t>& tokens) const;

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static TextGenerationFeaturizerPtr load(const std::string& filename);

  static TextGenerationFeaturizerPtr load_stream(std::istream& input_stream);

 private:
  // Private Constructor for Cereal
  TextGenerationFeaturizer() {}

  BoltVector lrcContext(const std::vector<uint32_t>& tokens,
                        uint32_t label_index) const;

  BoltVector ircContext(const std::vector<uint32_t>& tokens,
                        uint32_t label_index) const;

  BoltVector srcContext(const std::vector<uint32_t>& tokens,
                        uint32_t label_index) const;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Featurizer>(this), _lrc_len, _irc_len, _src_len);
  }
  /**
   * Helper function to featurize a single line from the text dataset and
   * returns the created input samples and labels.
   */
  std::vector<std::vector<BoltVector>> featurizeText(
      const std::string& line) const;

  static std::vector<uint32_t> parseTokens(const std::string& line);

  uint32_t _lrc_len;
  uint32_t _irc_len;
  uint32_t _src_len;
};

}  // namespace thirdai::dataset