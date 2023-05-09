#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <dataset/src/utils/TokenEncoding.h>
#include <utils/StringManipulation.h>
#include <string>

namespace thirdai::dataset {

class TextTokenizer {
 public:
  virtual std::vector<uint32_t> tokenize(const std::string& input) = 0;

  virtual std::string getResponsibleWord(const std::string& input,
                                         uint32_t source_token) = 0;

  virtual ~TextTokenizer() = default;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using TextTokenizerPtr = std::shared_ptr<TextTokenizer>;

class NaiveSplitTokenizer : public TextTokenizer {
 public:
  explicit NaiveSplitTokenizer(char delimiter = ' ') : _delimiter(delimiter) {}

  static auto make(char delimiter = ' ') {
    return std::make_shared<NaiveSplitTokenizer>(delimiter);
  }

  std::vector<uint32_t> tokenize(const std::string& input) final {
    return token_encoding::hashTokens(text::split(input, _delimiter));
  }

  std::string getResponsibleWord(const std::string& input,
                                 uint32_t source_token) final {
    auto map = token_encoding::buildUnigramHashToWordMap(
        text::split(input, _delimiter));

    if (!map.count(source_token)) {
      // should never get here since RCA should have only returned a valid token
      throw std::invalid_argument("Error in RCA.");
    }
    return map.at(source_token);
  }

 private:
  char _delimiter;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextTokenizer>(this), _delimiter);
  }
};

class WordPunctTokenizer : public TextTokenizer {
 public:
  WordPunctTokenizer() {}

  static auto make() { return std::make_shared<WordPunctTokenizer>(); }

  std::vector<uint32_t> tokenize(const std::string& input) final {
    return token_encoding::hashTokens(text::tokenizeSentence(input));
  }

  std::string getResponsibleWord(const std::string& input,
                                 uint32_t source_token) final {
    auto map = token_encoding::buildUnigramHashToWordMap(
        text::tokenizeSentence(input));

    if (!map.count(source_token)) {
      // should never get here since RCA should have only returned a valid token
      throw std::invalid_argument("Error in RCA.");
    }
    return map.at(source_token);
  }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextTokenizer>(this));
  }
};

class CharKGramTokenizer : public TextTokenizer {
 public:
  explicit CharKGramTokenizer(uint32_t k) : _k(k) {}

  static auto make(uint32_t k) {
    return std::make_shared<CharKGramTokenizer>(k);
  }

  std::vector<uint32_t> tokenize(const std::string& input) final {
    return token_encoding::hashTokens(text::charKGrams(input, _k));
  }

  std::string getResponsibleWord(const std::string& input,
                                 uint32_t source_token) final {
    auto map =
        token_encoding::buildUnigramHashToWordMap(text::charKGrams(input, _k));

    if (!map.count(source_token)) {
      // should never get here since RCA should have only returned a valid token
      throw std::invalid_argument("Error in RCA.");
    }
    return map.at(source_token);
  }

 private:
  uint32_t _k;

  CharKGramTokenizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextTokenizer>(this), _k);
  }
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::NaiveSplitTokenizer)
CEREAL_REGISTER_TYPE(thirdai::dataset::WordPunctTokenizer)
CEREAL_REGISTER_TYPE(thirdai::dataset::CharKGramTokenizer)