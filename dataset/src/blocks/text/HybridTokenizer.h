#pragma once

#include "TextTokenizer.h"
#include "WordpieceTokenizer.h"
#include <utils/CommonChecks.h>

namespace thirdai::dataset {

class HybridWordpieceCharKTokenizer : public TextTokenizer {
 public:
  explicit HybridWordpieceCharKTokenizer(const std::string& vocab_fpath,
                                         uint32_t k = 4,
                                         uint32_t wordpiece_range = 30000,
                                         uint32_t char_k_range = 70000,
                                         bool lowercase_wordpiece = true,
                                         bool lowercase_char_k = true)
      : _wordpiece_tokenizer(
            WordpieceTokenizer::make(vocab_fpath, lowercase_wordpiece)),
        _k(k),
        _wordpiece_range(wordpiece_range),
        _char_k_range(char_k_range),
        _lowercase_char_k(lowercase_char_k) {
    utils::validateGreaterThanZero(k, "k");
  }

  static auto make(const std::string& vocab_fpath, uint32_t k,
                   uint32_t wordpiece_range = 30000,
                   uint32_t char_k_range = 70000,
                   bool lowercase_wordpiece = true,
                   bool lowercase_char_k = true) {
    return std::make_shared<HybridWordpieceCharKTokenizer>(
        vocab_fpath, k, wordpiece_range, char_k_range, lowercase_wordpiece,
        lowercase_char_k);
  }

  std::vector<uint32_t> tokenize(const std::string& input) final {
    auto tokens = _wordpiece_tokenizer->tokenize(input);
    token_encoding::mod(tokens, _wordpiece_range);

    std::string possibly_lowercased_input = input;
    if (_lowercase_char_k) {
      possibly_lowercased_input = text::lower(possibly_lowercased_input);
    }

    auto char_k_tokens = token_encoding::hashTokens(text::wordLevelCharKGrams(
        text::tokenizeSentence(possibly_lowercased_input), /* k= */ _k,
        /* min_word_length= */ 1));

    token_encoding::mod(char_k_tokens, _char_k_range);

    for (uint32_t& char_k_token : char_k_tokens) {
      char_k_token += _wordpiece_range;
    }

    tokens.insert(tokens.end(), char_k_tokens.begin(),
                  char_k_tokens.end());

    return tokens;
  }

  std::string getResponsibleWord(const std::string& input,
                                 uint32_t source_token) final {
    (void)input;
    (void)source_token;
    throw std::invalid_argument("RCA not implemented for hybrid tokenizer.");
  }

 private:
  TextTokenizerPtr _wordpiece_tokenizer;
  uint32_t _k;
  uint32_t _wordpiece_range;
  uint32_t _char_k_range;
  bool _lowercase_char_k;

  HybridWordpieceCharKTokenizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextTokenizer>(this), _wordpiece_tokenizer, _k,
            _wordpiece_range, _char_k_range, _lowercase_char_k);
  }
};

using HybridWordpieceCharKTokenizerPtr =
    std::shared_ptr<HybridWordpieceCharKTokenizer>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::HybridWordpieceCharKTokenizer)