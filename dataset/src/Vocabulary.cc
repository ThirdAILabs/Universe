#include "Vocabulary.h"
#include "utils/StringManipulation.h"
#include <cassert>
#include <iostream>
#include <utf8proc.h>

namespace thirdai::dataset {

FixedVocabulary::FixedVocabulary(const std::string& file_path) {
  std::ifstream vocab_stream = SafeFileIO::ifstream(file_path);
  loadFromStream(vocab_stream);
}

FixedVocabulary::FixedVocabulary(std::istream& istream) {
  loadFromStream(istream);
}

void FixedVocabulary::loadFromStream(std::istream& vocab_stream) {
  // Add some special tokens before everything else.
  //
  // clang-tidy complains members should be initialized in initializer list,
  // unfortunately not possible without cruft (add is a non-static
  // member and expects the unordered map to be initialized).
  _unk_id = add(special_tokens::UNK);    // NOLINT
  _mask_id = add(special_tokens::MASK);  // NOLINT

  // Proceed to read from file to add the remaining vocabulary tokens. We
  // expect supplied files to be one token per-line.
  std::string vocab_token;
  while (getline(vocab_stream, vocab_token)) {
    add(vocab_token);
  }
}

uint32_t FixedVocabulary::size() const { return _token_to_id.size(); }

std::vector<uint32_t> FixedVocabulary::encode(
    const std::string_view& sentence) const {
  std::vector<uint32_t> token_ids;

  // The following describes a simple whitespace tokenization algorithm.
  // Multiple whitespaces are treated as a single separator. Any leading or
  // trailing whitespaces are discarded.
  const char* base = sentence.data();
  const char* end = base + sentence.size();
  const char* marker = base;

  // Advance marker until the next non-space character, also update base
  // to point accordingly - this is stripping leading spaces.
  while (marker != end && isspace(*marker)) {
    ++marker;
  }

  base = marker;

  while (marker != end) {
    if (isspace(*marker)) {
      // A word terminated by a space.
      size_t token_length = marker - base;
      std::string_view token(base, token_length);
      uint32_t token_id = id(token);
      token_ids.push_back(token_id);

      // Advance marker until the next non-space character, also update base
      // to point accordingly - strips trailing spaces and multiple spaces
      // between tokens.
      while (marker != end && isspace(*marker)) {
        ++marker;
      }
      base = marker;
    } else {
      ++marker;
    }
  }

  // There could be potential overhang, we cleave words only at detection of
  // space in the above loop. The overhang is detected to be a legit-token by
  // token length > 0.
  size_t token_length = marker - base;
  if (token_length) {
    std::string_view token(base, token_length);
    uint32_t token_id = id(token);
    token_ids.push_back(token_id);
  }

  return token_ids;
}

std::string FixedVocabulary::decode(
    const std::vector<uint32_t>& token_ids) const {
  std::stringstream stream;
  for (size_t i = 0; i < token_ids.size(); i++) {
    uint32_t token_id = token_ids[i];
    if (i != 0) {
      stream << " ";
    }
    auto query = _id_to_token.find(token_id);
    if (query != _id_to_token.end()) {
      std::string token = query->second;
      stream << token;
    } else {
      throw std::out_of_range(
          "Supplied token_ids contain out of bounds value.");
    }
  }

  return stream.str();
}

uint32_t FixedVocabulary::id(const std::string_view& token_view) const {
  std::string token(token_view.data(), token_view.size());
  auto query = _token_to_id.find(token);
  if (query == _token_to_id.end()) {
    return _unk_id;
  }

  uint32_t token_id = query->second;
  return token_id;
}

uint32_t FixedVocabulary::unkId() const { return _unk_id; }
uint32_t FixedVocabulary::maskId() const { return _mask_id; }

uint32_t FixedVocabulary::add(const std::string_view& token_view) {
  std::string token(token_view.data(), token_view.size());
  auto query = _token_to_id.find(token);
  if (query != _token_to_id.end()) {
    uint32_t token_id = query->second;
    return token_id;
  }

  uint32_t token_id = _token_to_id.size();
  _token_to_id.emplace(token, token_id);
  _id_to_token.emplace(token_id, token);
  return token_id;
}

Wordpiece::Vocab Wordpiece::load(const std::string& vocab_fpath) {
  Wordpiece::Vocab vocab;
  size_t index = 0;
  std::ifstream ifs(vocab_fpath, std::ifstream::in);
  std::string line;
  while (getline(ifs, line)) {
    std::wstring token = text::convertToUnicode(line);
    if (token.empty()) {
      break;
    }
    token = text::strip(token);
    vocab[token] = index;
    index++;
  }
  return vocab;
}

std::vector<std::wstring> Wordpiece::wordpieceTokenize(
    const std::wstring& text, const std::wstring& unkToken /*= L"[UNK]"*/,
    size_t maxInputCharsPerWord /*= 200*/) const {
  std::vector<std::wstring> outputTokens;
  for (const std::wstring& token : text::splitOnWhitespace(text)) {
    if (token.size() > maxInputCharsPerWord) {
      outputTokens.push_back(unkToken);
    }
    bool isBad = false;
    size_t start = 0;
    std::vector<std::wstring> subTokens;
    while (start < token.size()) {
      size_t end = token.size();
      std::wstring curSubstr;
      bool hasCurSubstr = false;
      while (start < end) {
        std::wstring substr;

        // Add ## prefix if we're in the middle of a word.
        if (start > 0) {
          substr += L"##";
        }

        substr += token.substr(start, end - start);

        if (_vocab.find(substr) != _vocab.end()) {
          curSubstr = substr;
          hasCurSubstr = true;
          break;
        }
        end--;
      }
      if (!hasCurSubstr) {
        isBad = true;
        break;
      }
      subTokens.push_back(curSubstr);
      start = end;
    }
    if (isBad) {
      outputTokens.push_back(unkToken);
    } else {
      outputTokens.insert(outputTokens.end(), subTokens.begin(),
                          subTokens.end());
    }
  }
  return outputTokens;
}

Wordpiece::Wordpiece(const std::string& vocab_fpath, bool to_lower)
    : _vocab(load(vocab_fpath)), _to_lower(to_lower) {
  for (auto& v : _vocab) {
    _inverse[v.second] = v.first;
  }
}

std::vector<std::wstring> Wordpiece::tokenize(const std::string& text) const {
  std::vector<std::wstring> splitTokens;
  for (auto& token : basicTokenize(text, _to_lower)) {
    for (auto& subToken : wordpieceTokenize(token)) {
      splitTokens.push_back(subToken);
    }
  }
  return splitTokens;
}
std::vector<uint32_t> Wordpiece::encode(
    const std::string_view& sentence) const {
  std::string sentence_copy(sentence.data(), sentence.size());
  std::vector<std::wstring> tokens = tokenize(sentence_copy);
  std::vector<uint32_t> ids = encodeTokens(tokens);
  return ids;
}

std::vector<uint32_t> Wordpiece::encodeTokens(
    const std::vector<std::wstring>& tokens) const {
  std::vector<uint32_t> ret(tokens.size());
  for (uint32_t i = 0; i < tokens.size(); i++) {
    auto query = _vocab.find(tokens[i]);
    assert(query != _vocab.end());
    ret[i] = query->second;
  }
  return ret;
}

uint32_t Wordpiece::id(const std::string_view& token_view) const {
  std::string token(token_view.data(), token_view.size());
  std::wstring wtoken = text::convertToUnicode(text::normalizeNFD(token));
  auto query = _vocab.find(wtoken);
  if (query != _vocab.end()) {
    return query->second;
  }
  return unkId();
}

uint32_t Wordpiece::size() const { return _vocab.size(); }

uint32_t Wordpiece::unkId() const {
  auto query = _vocab.find(L"[UNK]");
  assert(query != _vocab.end());
  return query->second;
}

uint32_t Wordpiece::maskId() const {
  auto query = _vocab.find(L"[MASK]");
  assert(query != _vocab.end());
  return query->second;
}

std::string Wordpiece::decode(const std::vector<uint32_t>& token_ids) const {
  std::string result;
  for (size_t i = 0; i < token_ids.size(); i++) {
    uint32_t token_id = token_ids[i];
    auto query = _inverse.find(token_id);
    assert(query != _inverse.end());

    std::string token = text::convertFromUnicode(query->second);
    bool is_subword_suffix = token.size() >= 2 && token.substr(0, 2) == "##";

    if (i != 0 and !is_subword_suffix) {
      result += " ";
    }

    result += is_subword_suffix ? token.substr(2) : token;
  }
  return result;
}

std::wstring Wordpiece::tokenizeChineseChars(const std::wstring& text) {
  std::wstring output;
  for (wchar_t c : text) {
    if (text::isChineseChar(c)) {
      output += L' ';
      output += c;
      output += L' ';
    } else {
      output += c;
    }
  }
  return output;
}

std::vector<std::wstring> Wordpiece::basicTokenize(const std::string& text,
                                                   bool to_lower) {
  std::wstring nText = text::convertToUnicode(text);
  nText = text::normalizeSpaces(nText);
  nText = tokenizeChineseChars(nText);

  const std::vector<std::wstring>& origTokens = text::splitOnWhitespace(nText);
  std::vector<std::wstring> splitTokens;
  for (std::wstring token : origTokens) {
    if (to_lower) {
      token = text::lower(token);
      token = text::stripAccents(token);
    }
    const auto& tokens = text::splitOnPunctuation(token);
    splitTokens.insert(splitTokens.end(), tokens.begin(), tokens.end());
  }
  return text::splitOnWhitespace(text::join(splitTokens, L" "));
}

}  // namespace thirdai::dataset
