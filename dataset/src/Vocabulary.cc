#include "Vocabulary.h"

namespace thirdai::dataset {

FixedVocabulary::FixedVocabulary(const std::string& file_path) {
  // Add some special tokens before everything else.
  //
  // clang-tidy complains members should be initialized in initializer list,
  // unfortunately not possible without cruft (add is a non-static
  // member and expects the unordered map to be initialized).
  _unk_id = add(special_tokens::UNK);    // NOLINT
  _mask_id = add(special_tokens::MASK);  // NOLINT

  // Proceed to read from file to add the remaining vocabulary tokens. We
  // expect supplied files to be one token per-line.
  std::ifstream vocab_stream = SafeFileIO::ifstream(file_path);
  std::string vocab_token;
  while (getline(vocab_stream, vocab_token)) {
    add(vocab_token);
  }
}

uint32_t FixedVocabulary::size() const { return _token_to_id.size(); }

std::vector<uint32_t> FixedVocabulary::encode(
    const std::string_view& sentence) const {
  std::vector<uint32_t> token_ids;

  const char* base = sentence.data();
  const char* end = base + sentence.size();
  const char* marker = base;
  while (marker != end) {
    if (isspace(*marker)) {
      // A word terminated by a space.
      size_t token_length = marker - base;
      std::string_view token(base, token_length);
      uint32_t token_id = id(token);
      token_ids.push_back(token_id);

      // Advance marker until the next non-space character, also update base
      // to point accordingly.
      while (marker != end && isspace(*marker)) {
        ++marker;
      }
      base = marker;
    } else {
      ++marker;
    }
  }

  // There could be potential overhang, we cleave words only at detection of
  // space in the above loop.
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

}  // namespace thirdai::dataset
