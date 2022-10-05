#pragma once

#include <dataset/src/utils/SafeFileIO.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

namespace special_tokens {
constexpr std::string_view UNK = "[UNK]";
constexpr std::string_view MASK = "[MASK]";
}  // namespace special_tokens

/*
 * The following class specifies the vocabulary public interface for language
 * tasks. The vocabulary is intended to provide numerical ids to string tokens
 * (encoding) and obtain the inverse mapping for decoding.
 */
class Vocabulary {
 public:
  // Encode given sentence (sequence of tokens) into numerical ids assigned by
  // the vocabulary.
  virtual std::vector<uint32_t> encode(
      const std::string_view& sentence) const = 0;

  // Decodes given piece_ids into a string. Throws out of bounds exception if
  // piece outside what's known to the vocabulary.
  virtual std::string decode(const std::vector<uint32_t>& piece_ids) const = 0;

  // Returns the id of a given token, if it exists in the vocabulary.
  // If token not present, returns unkId, indicating the token is unknown to the
  // vocabulary.
  virtual uint32_t id(const std::string_view& token_view) const = 0;

  // Returns the total size of the vocabulary.
  virtual uint32_t size() const = 0;

  // Returns the id of unknown token.
  virtual uint32_t unkId() const = 0;

  // Returns id of mask special token.
  virtual uint32_t maskId() const = 0;

  // To satisfy clang
  virtual ~Vocabulary() = default;
};

class FixedVocabulary : public Vocabulary {
 public:
  // Construct vocabulary from a given file. The file is expected to contain
  // each unique token in a line. The initial ids are assigned to special
  // tokens, and the tokens in file are read.
  explicit FixedVocabulary(const std::string& file_path);

  uint32_t size() const final;

  std::vector<uint32_t> encode(const std::string_view& sentence) const final;

  std::string decode(const std::vector<uint32_t>& piece_ids) const final;

  uint32_t id(const std::string_view& token_view) const final;

  uint32_t unkId() const final;

  uint32_t maskId() const final;

  static std::shared_ptr<Vocabulary> make(const std::string& file_path) {
    return std::make_shared<FixedVocabulary>(file_path);
  }

 private:
  // Stores the forward map from string-token to uint32_t ids.
  std::unordered_map<std::string, uint32_t> _forward;

  // Stores the inverse map from uint32_t id to token. Useful when needed for
  // decoding.
  std::unordered_map<uint32_t, std::string> _backward;

  uint32_t _unk_id, _mask_id;

  // Does not check if token already exist, directly adds. This saves some
  // compute when we know there cannot be duplicates by construction.
  uint32_t add(const std::string_view& token_view);
};

}  // namespace thirdai::dataset
