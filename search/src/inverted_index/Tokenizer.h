#pragma once

#include <archive/src/Archive.h>
#include <utils/text/PorterStemmer.h>
#include <utils/text/StringManipulation.h>
#include <memory>
#include <optional>
#include <vector>

namespace thirdai::search {

using Token = std::string;
using Tokens = std::vector<Token>;

class Tokenizer {
 public:
  virtual Tokens tokenize(const std::string& text) const = 0;

  virtual ar::ConstArchivePtr toArchive() const = 0;

  static std::shared_ptr<Tokenizer> fromArchive(const ar::Archive& archive);

  virtual ~Tokenizer() = default;
};

using TokenizerPtr = std::shared_ptr<Tokenizer>;

class DefaultTokenizer final : public Tokenizer {
 public:
  explicit DefaultTokenizer(bool stem = true, bool lowercase = true)
      : _stem(stem), _lowercase(lowercase) {}

  Tokens tokenize(const std::string& input) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "default"; }

  static std::shared_ptr<DefaultTokenizer> fromArchive(
      const ar::Archive& archive);

 private:
  bool _stem, _lowercase;
};

/**
 * This tokenizer splits a string into words delimited by spaces and punctuation
 * marks, then outputs k-grams within each word.
 * The behavior enabled by the soft_start flag is best explained with an
 * example:
 * When soft_start is false,
 * tokenize("chanel") -> ["chan", "hane", "anel"]
 * When soft_start is true,
 * tokenize("chanel") -> ["c", "ch", "cha", "chan", "hane", "anel"]
 * This feature is useful for autocomplete use cases.
 */
class WordKGrams final : public Tokenizer {
 public:
  explicit WordKGrams(uint32_t k = 4, bool soft_start = true,
                      bool include_whole_words = true, bool stem = true,
                      bool lowercase = true)
      : _k(k),
        _soft_start(soft_start),
        _include_whole_words(include_whole_words),
        _stem(stem),
        _lowercase(lowercase) {}

  Tokens tokenize(const std::string& input) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "word_k_grams"; }

  static std::shared_ptr<WordKGrams> fromArchive(const ar::Archive& archive);

 private:
  uint32_t _k;
  bool _soft_start, _include_whole_words, _stem, _lowercase;
};

}  // namespace thirdai::search