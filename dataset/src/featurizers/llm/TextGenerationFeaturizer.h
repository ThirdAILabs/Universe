#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/featurizers/llm/TextContextFeaturizer.h>
#include <json/include/nlohmann/json.hpp>
#include <limits>
#include <stdexcept>

using json = nlohmann::json;

namespace thirdai::dataset {

/**
 * Featurizes a list of rows from a text dataset for next word prediction.
 * Expects that each line will be a json object containing the fields 'target'
 * and optional 'prompt' or 'context'. Each of these fields should be a string
 * containing a list of space separate bert tokens (integer ids) for the text of
 * a given document/text. Returns a list of the input samples and labels for the
 * documents/texts in the input rows.
 *
 * The tokens in the 'target' field are featurized into 3 parts. The long range
 * context (lrc), the intermediate range context (irc), and the short range
 * context (src). The long range context will be featurized using unigrams, the
 * intermediate range context will be featurized using pairgrams, and the short
 * range context will be featurized with unigrams and padded to a constistent
 * length, this is so these token embeddings can be concatenated in the text
 * generation model.
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
 * If the 'context' field is specified then the context is prepended to the
 * target tokens, however the generation begins at the begining of the target
 * tokens. The context tokens will be included in the lrc, irc, and src contexts
 * but will not be used as labels during training.
 *
 * Note: note that the first token is assumed to be [CLS], and the token 0
 * denotes [PAD].
 *
 * The prompt field indicates which tokens to return as the prompt inputs to the
 * model. If no prompt field is specified, then a single padding token is
 * returned.
 *
 * The featurizer returns 5 inputs to the model:
 *     (prompt), (lrc context), (irc context), (src context), (label)
 */
class TextGenerationFeaturizer;
using TextGenerationFeaturizerPtr = std::shared_ptr<TextGenerationFeaturizer>;

class TextGenerationFeaturizer final : public Featurizer {
 public:
  TextGenerationFeaturizer(uint32_t lrc_len, uint32_t irc_len, uint32_t src_len,
                           uint32_t vocab_size, bool include_position = false,
                           bool featurize_in_chunks = true)
      : _context_featurizer(lrc_len, irc_len, src_len, vocab_size,
                            include_position),
        _featurize_in_chunks(featurize_in_chunks) {
    if (irc_len > lrc_len || src_len > lrc_len) {
      throw std::invalid_argument("LRC size should be at least IRC/SRC size.");
    }
  }

  std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& lines) final;

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

  size_t getNumDatasets() final { return 5; }

  std::vector<uint32_t> getDimensions() final {
    throw std::runtime_error(
        "getDimensions is not supported for TextGenerationFeaturizer.");
  }

  // There is no target because we are only making a single prediction at the
  // end of the context, and thus no need for a set of target tokens.
  std::vector<BoltVector> featurizeInferenceSample(
      const std::vector<uint32_t>& prompt,
      const std::vector<uint32_t>& context) const;

  bolt::TensorList featurizeInputBatch(
      const std::vector<std::vector<uint32_t>>& tokens,
      const std::vector<uint32_t>& dims) const;

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static TextGenerationFeaturizerPtr load(const std::string& filename);

  static TextGenerationFeaturizerPtr load_stream(std::istream& input_stream);

 private:
  // Private Constructor for Cereal
  TextGenerationFeaturizer() {}

  static BoltVector promptContext(const std::vector<uint32_t>& prompt_tokens);

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Featurizer>(this), _context_featurizer);
  }

  /**
   * Helper function to featurize a single line from the text dataset and
   * returns the created input samples and labels.
   */
  std::vector<std::vector<BoltVector>> featurizeTextChunks(
      const std::string& line) const;

  std::vector<std::vector<BoltVector>> featurizeTextSlidingWindow(
      const std::string& line) const;

  /**
   * Returns the context tokens (the concatenation of the context and target) as
   * well as the index to start predicting from.
   */
  static std::pair<std::vector<uint32_t>, size_t> getAllTokens(
      const json& line_content, bool with_context);

  static std::vector<uint32_t> getPrompt(const json& line_content);

  TextContextFeaturizer _context_featurizer;
  bool _featurize_in_chunks;
};

}  // namespace thirdai::dataset