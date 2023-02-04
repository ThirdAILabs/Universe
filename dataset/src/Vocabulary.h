#pragma once

#include <dataset/src/utils/SafeFileIO.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utf8proc.h>
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
 *
 * Unless marked explicitly here otherwise - API interface should be considered
 * unstable, and is expected to stabilize across experiments.
 */

class Vocabulary {
 public:
  // Encode given sentence (sequence of tokens) into numerical ids assigned by
  // the vocabulary.
  virtual std::vector<uint32_t> encode(
      const std::string_view& sentence) const = 0;

  // Decodes given token_ids into a string. Throws out of bounds exception if
  // piece outside what's known to the vocabulary.
  virtual std::string decode(const std::vector<uint32_t>& token_ids) const = 0;

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
  // Construct a new Fixed Vocabulary object from a given file-path. The
  // file is expected to contain each unique token in a line. The initial ids
  // are assigned to special tokens, and the tokens in file are read.
  explicit FixedVocabulary(const std::string& file_path);

  // Alternate constructor to be able to load from an istream. Exposed
  // for the sole purpose of testing.
  explicit FixedVocabulary(std::istream& istream);

  // Number of unique tokens constituting the vocabulary.
  uint32_t size() const final;

  // Encode a sentence into vector of integral vocab-ids inferred from tokens.
  // Whitespace tokenization under the hood, stripping leading, trailing
  // spaces. See decode(...) for inverse operation.
  std::vector<uint32_t> encode(const std::string_view& sentence) const final;

  // Decode a sequence of integral vocab-ids into equivalent string
  // representation. If vocab-id not known (i.e >= size()), out of range
  // exception is thrown. Inverse of encode(...).
  std::string decode(const std::vector<uint32_t>& token_ids) const final;

  // Get id of a token if in vocabulary. Returns unkId() is token not known to
  // vocabulary.
  uint32_t id(const std::string_view& token_view) const final;

  // Get id corresponding to the unknown special token.
  uint32_t unkId() const final;

  // Get id corresponding to the mask special token.
  uint32_t maskId() const final;

  // Create a generic Vocabulary given a file-path. Useful in python, to
  // minimize having to export the functions repeatedly by binding.
  static std::shared_ptr<Vocabulary> make(const std::string& file_path) {
    return std::make_shared<FixedVocabulary>(file_path);
  }

 private:
  // Load a vocabulary from a file-stream. Common code shared across both
  // constructors (from filepath and from stream).
  void loadFromStream(std::istream& vocab_stream);

  // Stores the forward map from string-token to uint32_t ids.
  std::unordered_map<std::string, uint32_t> _token_to_id;

  // Stores the inverse map from uint32_t id to token. Useful when needed for
  // decoding.
  std::unordered_map<uint32_t, std::string> _id_to_token;

  // Basic special token ids - for unknown and mask, stored explicitly.
  uint32_t _unk_id, _mask_id;

  // Does not check if token already exist, directly adds. This saves some
  // compute when we know there cannot be duplicates by construction.
  uint32_t add(const std::string_view& token_view);
};

class Basic {
 public:
  explicit Basic(bool lower_case = true);
  std::vector<std::wstring> tokenize(const std::string& text) const;

 private:
  std::wstring cleanText(const std::wstring& text) const;
  std::wstring tokenizeChineseChars(const std::wstring& text) const;
  std::wstring strip(const std::wstring& text) const;
  std::vector<std::wstring> split(const std::wstring& text) const;
  std::wstring runStripAccents(const std::wstring& text) const;
  std::vector<std::wstring> runSplitOnPunc(const std::wstring& text) const;

  bool _to_lower;
};

class Wordpiece {
 public:
  explicit Wordpiece(const std::string& vocabFile, bool lower_case = true);
  std::vector<std::wstring> tokenize(const std::string& text) const;
  std::vector<size_t> encode(const std::vector<std::wstring>& text) const;
  std::vector<std::wstring> wordpiece_tokenize(
      const std::wstring& text, const std::wstring& unkToken = L"[UNK]",
      size_t maxInputCharsPerWord = 200) const;

 private:
  using Vocab = std::unordered_map<std::wstring, size_t>;
  using InvVocab = std::unordered_map<size_t, std::wstring>;
  static Vocab loadVocab(const std::string& vocabFile);
  Vocab _vocab;
  InvVocab _inverse;
  std::string _vocab_fpath;
  Basic _basic;
};

namespace detail {

const std::wstring DEFAULT_STRIP_CHARACTERS = L" \t\n\r\v\f";

class is_any_of {
 public:
  explicit is_any_of(const std::wstring& delimiters);
  bool operator()(wchar_t candidate) const;

 private:
  std::wstring delimiters_;
};

std::wstring join(const std::vector<std::wstring>& atoms,
                  const std::wstring& delimiter);

template <class Predicate>
void split(std::vector<std::wstring>& result, const std::wstring& s,
           Predicate predicate);
//
std::string convertFromUnicode(const std::wstring& wText);
std::wstring convertToUnicode(const std::string& text);
std::string normalize_nfd(const std::string& s);
std::wstring tolower(const std::wstring& s);

bool isControl(const wchar_t& ch);
bool isWhitespace(const wchar_t& ch);
bool isPunctuation(const wchar_t& ch);
bool isChineseChar(const wchar_t& ch);
bool isStripChar(const wchar_t& ch);
}  // namespace detail

}  // namespace thirdai::dataset
