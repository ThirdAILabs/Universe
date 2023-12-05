#pragma once

#include "TextTokenizer.h"
#include "WordpieceTokenizer.h"

namespace thirdai::dataset {
class HybridWordpieceCharKTokenizer : public TextTokenizer {
 public:
  explicit HybridWordpieceCharKTokenizer(
      WordpieceTokenizerPtr wordpiece_tokenizer, uint32_t k)
      : _wordpiece_tokenizer(std::move(wordpiece_tokenizer)), _k(k) {}

  static auto make(const WordpieceTokenizerPtr& wordpiece_tokenizer,
                   uint32_t k) {
    return std::make_shared<HybridWordpieceCharKTokenizer>(wordpiece_tokenizer,
                                                           k);
  }

  std::vector<uint32_t> tokenize(const std::string& input) final {
    auto tokens = token_encoding::hashTokens(
        text::wordLevelCharKGrams(text::tokenizeSentence(input), _k, 3));
    token_encoding::mod(tokens, 70000);

    auto wordpiece_tokens = _wordpiece_tokenizer->tokenize(input);
    token_encoding::mod(wordpiece_tokens, 30000);
    for (uint32_t& wordpiece_token : wordpiece_tokens) {
      wordpiece_token += 70000;
    }

    tokens.insert(tokens.end(), wordpiece_tokens.begin(),
                  wordpiece_tokens.end());
    return tokens;
  }

  std::string getResponsibleWord(const std::string& input,
                                 uint32_t source_token) final {
    auto map =
        token_encoding::buildUnigramHashToWordMap(text::charKGrams(input, _k));

    if (!map.count(source_token)) {
      // should never get here since RCA should have only returned a valid token
      throw std::invalid_argument("Error in RCA.");
    }
    return map.at(source_token);
  }

 private:
  WordpieceTokenizerPtr _wordpiece_tokenizer;
  uint32_t _k;

  HybridWordpieceCharKTokenizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextTokenizer>(this), _wordpiece_tokenizer, _k);
  }
};

using HybridWordpieceCharKTokenizerPtr =
    std::shared_ptr<HybridWordpieceCharKTokenizer>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::HybridWordpieceCharKTokenizer)