#include "WordpieceTokenizer.h"
#include <archive/src/Archive.h>
#include <utils/StringManipulation.h>
#include <cassert>
#include <iostream>

namespace thirdai::dataset {

WordpieceTokenizer::WordpieceTokenizer(const std::string& vocab_fpath,
                                       bool to_lower)
    : _to_lower(to_lower) {
  size_t token_id = 0;
  std::ifstream vocab_stream = SafeFileIO::ifstream(vocab_fpath);
  std::string line;
  while (getline(vocab_stream, line)) {
    std::wstring token = text::toUnicode(line);
    if (token.empty()) {
      break;
    }
    token = text::strip(token);
    _token_to_id[token] = token_id;
    _id_to_token.push_back(token);
    token_id++;
  }

  if (!_token_to_id.count(std::wstring(special_tokens::UNK))) {
    _token_to_id[std::wstring(special_tokens::UNK)] = _token_to_id.size();
    _id_to_token.push_back(std::wstring(special_tokens::UNK));
  }

  if (!_token_to_id.count(std::wstring(special_tokens::MASK))) {
    _token_to_id[std::wstring(special_tokens::MASK)] = _token_to_id.size();
    _id_to_token.push_back(std::wstring(special_tokens::MASK));
  }
}

std::vector<uint32_t> WordpieceTokenizer::tokenize(
    const std::string& sentence) {
  std::vector<std::wstring> tokens = tokenizeToStrings(sentence);
  std::vector<uint32_t> encoded(tokens.size());
  for (uint32_t i = 0; i < tokens.size(); i++) {
    auto query = _token_to_id.find(tokens[i]);
    assert(query != _token_to_id.end());
    encoded[i] = query->second;
  }
  return encoded;
}

std::string WordpieceTokenizer::decode(
    const std::vector<uint32_t>& token_ids) const {
  std::string result;
  for (size_t i = 0; i < token_ids.size(); i++) {
    uint32_t token_id = token_ids[i];
    if (token_id >= _id_to_token.size()) {
      throw std::invalid_argument("Attempting to decode invalid token: " +
                                  std::to_string(token_id) + ".");
    }

    std::string token = text::fromUnicode(_id_to_token.at(token_id));
    bool is_subword_suffix = token.size() >= 2 && token.substr(0, 2) == "##";

    if (i != 0 and !is_subword_suffix) {
      result += " ";
    }

    result += is_subword_suffix ? token.substr(2) : token;
  }
  return result;
}

uint32_t WordpieceTokenizer::id(const std::string& token) const {
  std::wstring wtoken = text::toUnicode(text::normalize(token));
  auto query = _token_to_id.find(wtoken);
  if (query != _token_to_id.end()) {
    return query->second;
  }
  return unkId();
}

std::vector<std::wstring> WordpieceTokenizer::tokenizeToStrings(
    const std::string& sentence) const {
  std::vector<std::wstring> subwords;

  std::vector<std::wstring> tokens = basicTokenize(sentence, _to_lower);
  for (const std::wstring& token : tokens) {
    std::vector<std::wstring> wordpieces = wordpieceTokenize(token);
    subwords.insert(subwords.end(), wordpieces.begin(), wordpieces.end());
  }

  return subwords;
}

std::vector<std::wstring> WordpieceTokenizer::basicTokenize(
    const std::string& text, bool to_lower) {
  std::wstring u_text = text::toUnicode(text::normalize(text));
  u_text = text::cleanText(u_text);
  u_text = tokenizeChineseChars(u_text);

  std::vector<std::wstring> tokens;

  // Split at (normalized) whitespaces to begin with.
  std::vector<std::wstring> space_tokens = text::splitOnWhitespace(u_text);
  for (std::wstring space_token : space_tokens) {
    if (to_lower) {
      space_token = text::lower(space_token);
    }

    // We strip accents by default. TODO(david): Reevaluate an option for this
    // for other languages. This is done by not removing utf8proc category ==
    // UTF8PROC_CATEGORY_MN in text::cleanText(..)

    // Tokenize by punctuations. This means punctuations appear in text as
    // tokens.
    std::vector<std::wstring> punct_tokens =
        text::tokenizeByPunctuations(space_token);

    tokens.insert(tokens.end(), punct_tokens.begin(), punct_tokens.end());
  }

  return tokens;
}

std::wstring WordpieceTokenizer::tokenizeChineseChars(
    const std::wstring& text) {
  std::wstring output;
  for (const wchar_t c : text) {
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

std::vector<std::wstring> WordpieceTokenizer::wordpieceTokenize(
    const std::wstring& text, const std::wstring& unk /*= L"[UNK]"*/,
    size_t max_chars_per_wordpiece /*= 200*/) const {
  // This is mostly from
  // https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/models/bert/tokenization_bert.py#L512-L558
  std::vector<std::wstring> wordpieces;

  std::vector<std::wstring> words = text::splitOnWhitespace(text);
  for (const std::wstring& token : words) {
    if (token.size() > max_chars_per_wordpiece) {
      wordpieces.push_back(unk);
      continue;
    }

    std::vector<std::wstring> subwords;

    bool is_bad = false;
    size_t start = 0;

    while (start < token.size()) {
      // Add ## prefix if we're in the middle of a word.
      std::wstring candidate =
          start > 0 ? L"##" + token.substr(start) : token.substr(start);
      size_t min_candidate_size = start > 0 ? 2 : 0;
      bool candidate_valid = false;
      while (candidate.size() > min_candidate_size) {
        if (_token_to_id.find(candidate) != _token_to_id.end()) {
          candidate_valid = true;
          break;
        }
        candidate.pop_back();
      }
      if (!candidate_valid) {
        is_bad = true;
        break;
      }
      subwords.push_back(candidate);
      start += (start > 0 ? candidate.size() - 2 : candidate.size());
    }

    if (is_bad) {
      wordpieces.push_back(unk);
    } else {
      wordpieces.insert(wordpieces.end(), subwords.begin(), subwords.end());
    }
  }

  return wordpieces;
}

ar::ConstArchivePtr WordpieceTokenizer::toArchive() const {
  auto map = ar::Map::make();
  map->set("type", ar::str(type()));
  map->set("id_to_token", ar::vecWStr(_id_to_token));
  map->set("to_lower", ar::boolean(_to_lower));
  return map;
}

WordpieceTokenizer::WordpieceTokenizer(const ar::Archive& archive)
    : _id_to_token(archive.getAs<ar::VecWStr>("id_to_token")),
      _to_lower(archive.getAs<ar::Boolean>("to_lower")) {
  for (size_t i = 0; i < _id_to_token.size(); i++) {
    _token_to_id[_id_to_token[i]] = i;
  }
}

}  // namespace thirdai::dataset
