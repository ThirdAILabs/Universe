#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "TextTokenizer.h"
#include <dataset/src/utils/SafeFileIO.h>
#include <proto/tokenizers.pb.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

namespace special_tokens {
constexpr std::wstring_view UNK = L"[UNK]";
constexpr std::wstring_view MASK = L"[MASK]";
}  // namespace special_tokens

class WordpieceTokenizer final : public TextTokenizer {
 public:
  explicit WordpieceTokenizer(const std::string& vocab_fpath,
                              bool to_lower = true);

  explicit WordpieceTokenizer(const proto::data::WordpieceTokenizer& wordpiece);

  static std::shared_ptr<TextTokenizer> make(const std::string& vocab_file,
                                             bool lowercase = true) {
    return std::make_shared<WordpieceTokenizer>(vocab_file, lowercase);
  }

  std::vector<uint32_t> tokenize(const std::string& sentence) final;

  std::vector<std::wstring> tokenizeToStrings(
      const std::string& sentence) const;

  std::string decode(const std::vector<uint32_t>& token_ids) const;

  uint32_t id(const std::string& token) const;

  uint32_t size() const { return _token_to_id.size(); }

  uint32_t unkId() const { return _token_to_id.at(L"[UNK]"); }

  uint32_t maskId() const { return _token_to_id.at(L"[MASK]"); };

  std::string getResponsibleWord(const std::string& input,
                                 uint32_t source_token) final {
    (void)input;
    // TODO(david): should we take the whole word here instead of the subword
    return decode({source_token});
  }

  proto::data::Tokenizer* toProto() const final;

 private:
  /**
   * This function handles a lot of the preprocessing that is needed before we
   * can tokenize via subwords (calling wordpieceTokenize). Mainly this
   * include splitting by white space/punctuations, converting to unicode,
   * normalizing spaces, separating out chinese special characters, etc.
   */
  static std::vector<std::wstring> basicTokenize(const std::string& text,
                                                 bool to_lower);

  std::vector<std::wstring> wordpieceTokenize(
      const std::wstring& text, const std::wstring& unk = L"[UNK]",
      size_t max_chars_per_wordpiece = 200) const;

  static std::wstring tokenizeChineseChars(const std::wstring& text);

  using TokenToId = std::unordered_map<std::wstring, size_t>;

  // private constructor for cereal
  WordpieceTokenizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextTokenizer>(this), _token_to_id, _id_to_token,
            _to_lower);
  }

  TokenToId _token_to_id;
  std::vector<std::wstring> _id_to_token;

  bool _to_lower;
};

using WordpieceTokenizerPtr = std::shared_ptr<WordpieceTokenizer>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::WordpieceTokenizer)
