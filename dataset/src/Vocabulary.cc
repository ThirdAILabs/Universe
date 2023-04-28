#include "Vocabulary.h"
#include <utils/StringManipulation.h>
#include <iostream>

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

WordpieceVocab::WordpieceVocab(const std::string& vocab_fpath, bool to_lower)
    : _token_to_id(load(vocab_fpath)), _to_lower(to_lower) {
  for (const auto& [token, id] : _token_to_id) {
    _id_to_token[id] = token;
  }
}

WordpieceVocab::TokenToId WordpieceVocab::load(const std::string& vocab_fpath) {
  WordpieceVocab::TokenToId vocab;
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

std::vector<uint32_t> WordpieceVocab::encode(
    const std::string_view& sentence) const {
  std::string buffer(sentence.data(), sentence.size());
  std::vector<std::wstring> tokens = tokenize(buffer);
  std::vector<uint32_t> encoded(tokens.size());
  for (uint32_t i = 0; i < tokens.size(); i++) {
    auto query = _word_to_id.find(tokens[i]);
    assert(query != _word_to_id.end());
    encoded[i] = query->second;
  }
  return encoded;
}

std::string WordpieceVocab::decode(
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

uint32_t WordpieceVocab::id(const std::string_view& token_view) const {
  std::string token(token_view.data(), token_view.size());
  std::wstring wtoken = text::toUnicode(token);  // TODO(david) normalize here?
  auto query = _token_to_id.find(wtoken);
  if (query != _token_to_id.end()) {
    return query->second;
  }
  return unkId();
}

}  // namespace thirdai::dataset
