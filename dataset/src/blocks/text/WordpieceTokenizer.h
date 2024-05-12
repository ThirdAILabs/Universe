#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "TextTokenizer.h"
#include <dataset/src/utils/SafeFileIO.h>
#include <codecvt>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

inline std::string wstring_to_string(const std::wstring& wstr) {
  // This is used to transform wstring tokens to UTF-8 tokens.
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return converter.to_bytes(wstr);
}

namespace special_tokens {
constexpr std::wstring_view UNK = L"[UNK]";
constexpr std::wstring_view MASK = L"[MASK]";
}  // namespace special_tokens

class WordpieceTokenizer : public TextTokenizer {
 public:
  explicit WordpieceTokenizer(const std::string& vocab_fpath,
                              bool to_lower = true);

  explicit WordpieceTokenizer(const ar::Archive& archive);

  static std::shared_ptr<TextTokenizer> make(const std::string& vocab_file,
                                             bool lowercase = true) {
    return std::make_shared<WordpieceTokenizer>(vocab_file, lowercase);
  }

  std::vector<uint32_t> tokenize(const std::string& sentence) final;

  std::vector<std::wstring> tokenizeToStrings(
      const std::string& sentence) const;

  std::vector<std::string> toStrings(const std::string& input) final {
    auto wstring_tokens = tokenizeToStrings(input);
    std::vector<std::string> str_tokens;
    str_tokens.reserve(wstring_tokens.size());

    for (const auto& tok : wstring_tokens) {
      str_tokens.emplace_back(wstring_to_string(tok));
    }
    return str_tokens;
  }

  std::string decode(const std::vector<uint32_t>& token_ids) const;

  uint32_t id(const std::string& token) const;

  std::string token(uint32_t id) const;

  uint32_t size() const { return _token_to_id.size(); }

  size_t vocabSize() const { return _id_to_token.size(); }

  uint32_t unkId() const { return _token_to_id.at(L"[UNK]"); }

  uint32_t maskId() const { return _token_to_id.at(L"[MASK]"); };

  std::string getResponsibleWord(const std::string& input,
                                 uint32_t source_token) final {
    (void)input;
    // TODO(david): should we take the whole word here instead of the subword
    return decode({source_token});
  }

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "wordpiece"; }

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
