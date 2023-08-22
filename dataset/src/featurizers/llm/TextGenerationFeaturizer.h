#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/featurizers/llm/TextContextFeaturizer.h>
#include <json/include/nlohmann/json.hpp>
#include <limits>
#include <stdexcept>

using json = nlohmann::json;

namespace thirdai::dataset {


class TextGenerationFeaturizer;
using TextGenerationFeaturizerPtr = std::shared_ptr<TextGenerationFeaturizer>;

class TextGenerationFeaturizer final : public Featurizer {
 public:
  TextGenerationFeaturizer(uint32_t lrc_len, uint32_t irc_len, uint32_t src_len,
                           uint32_t vocab_size, bool include_position = false)
      : _context_featurizer(lrc_len, irc_len, src_len, vocab_size,
                            include_position) {
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

  static std::string getStringField(const json& json_object, const std::string& name) {
  if (!json_object[name].is_string()) {
    throw std::invalid_argument("Expected field '" + name +
                                "' to be a string.");
  }
  return json_object[name].get<std::string>();
}
  // There is no target because we are only making a single prediction at the
  // end of the context, and thus no need for a set of target tokens.
  std::vector<BoltVector> featurizeInferenceSample(
      const std::vector<uint32_t>& prompt,
      const std::vector<uint32_t>& context) const;

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
  std::vector<std::vector<BoltVector>> featurizeText(
      const std::string& line) const;

  /**
   * Returns the context tokens (the concatenation of the context and target) as
   * well as the index to start predicting from.
   */
  static std::vector<uint32_t> getAllTokens(const json& line_content);

  static std::vector<uint32_t> getPrompt(const json& line_content);

  TextContextFeaturizer _context_featurizer;
};

}  // namespace thirdai::dataset