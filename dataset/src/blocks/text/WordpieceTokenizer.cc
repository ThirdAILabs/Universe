#include "WordpieceTokenizer.h"
#include <utils/StringManipulation.h>
#include <iostream>

namespace thirdai::dataset {

WordpieceTokenizer::WordpieceTokenizer(const std::string& vocab_fpath,
                                       bool to_lower)
    : _token_to_id(load(vocab_fpath)), _to_lower(to_lower) {
  for (const auto& [token, id] : _token_to_id) {
    _id_to_token[id] = token;
  }
}

WordpieceTokenizer::TokenToId WordpieceTokenizer::load(
    const std::string& vocab_fpath) {
  WordpieceTokenizer::TokenToId vocab;
  size_t token_id = 0;
  std::ifstream vocab_stream = SafeFileIO::ifstream(vocab_fpath);
  std::string line;
  while (getline(vocab_stream, line)) {
    std::wstring token = text::toUnicode(line);
    if (token.empty()) {
      break;
    }
    token = text::strip(token);
    vocab[token] = token_id;
    token_id++;
  }
  return vocab;
}

std::vector<uint32_t> WordpieceTokenizer::tokenize(
    const std::string& sentence) {
  std::string buffer(sentence.data(), sentence.size());
  std::vector<std::wstring> tokens = tokenizeToStrings(buffer);
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
    if (!_id_to_token.count(token_id)) {
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

uint32_t WordpieceTokenizer::id(const std::string& token_view) const {
  std::string token(token_view.data(), token_view.size());
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
  std::wstring u_text = text::toUnicode(text);
  u_text = text::normalizeSpaces(u_text);
  u_text = tokenizeChineseChars(u_text);

  std::vector<std::wstring> tokens;

  // Split at (normalized) whitespaces to begin with.
  std::vector<std::wstring> space_tokens = text::splitOnWhitespace(u_text);
  for (std::wstring space_token : space_tokens) {
    if (to_lower) {
      space_token = text::lower(space_token);
      space_token = text::stripAccents(space_token);
    }

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

  // TODO(jerin-thirdai): The following block looks like it can be simplified.
  // It is currently riddled with jump statements and can be more structured.
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
      size_t end = token.size();
      std::wstring candidate;
      bool candidate_valid = false;
      while (start < end) {
        std::wstring buffer;

        // Add ## prefix if we're in the middle of a word.
        if (start > 0) {
          buffer += L"##";
        }

        buffer += token.substr(start, end - start);

        if (_token_to_id.find(buffer) != _token_to_id.end()) {
          candidate = buffer;
          candidate_valid = true;
          break;
        }

        end--;
      }

      if (!candidate_valid) {
        is_bad = true;
        break;
      }

      subwords.push_back(candidate);
      start = end;
    }

    if (is_bad) {
      wordpieces.push_back(unk);
    } else {
      wordpieces.insert(wordpieces.end(), subwords.begin(), subwords.end());
    }
  }
  return wordpieces;
}

}  // namespace thirdai::dataset
