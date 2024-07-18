#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <utils/text/StringManipulation.h>
#include <string>

namespace thirdai::dataset {

class TextTokenizer {
 public:
  virtual std::vector<uint32_t> tokenize(const std::string& input) = 0;

  virtual std::string getResponsibleWord(const std::string& input,
                                         uint32_t source_token) = 0;

  virtual ar::ConstArchivePtr toArchive() const = 0;

  virtual std::vector<std::string> toStrings(const std::string& input) = 0;

  static std::shared_ptr<TextTokenizer> fromArchive(const ar::Archive& archive);

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

  std::vector<std::string> toStrings(const std::string& input) final {
    return text::split(input, _delimiter);
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

  ar::ConstArchivePtr toArchive() const final {
    auto map = ar::Map::make();
    map->set("type", ar::str(type()));
    map->set("delimiter", ar::character(_delimiter));
    return map;
  }

  static std::string type() { return "naive_split"; }

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

  std::vector<std::string> toStrings(const std::string& input) final {
    return text::tokenizeSentence(input);
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

  ar::ConstArchivePtr toArchive() const final {
    auto map = ar::Map::make();
    map->set("type", ar::str(type()));
    return map;
  }

  static std::string type() { return "word_punct"; }

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
    auto tokens = text::charKGrams(input, _k);
    std::cout << "Char-4 Tokens" << std::endl;
    for(const auto &token: tokens){
      std::cout << token << std::endl;
    }
    std::cout << "Char-4 Tokens" << std::endl;
    auto hash_tokens = token_encoding::hashTokens(tokens);
    std::cout << "Hash Tokens: " << std::endl;
    for(const auto & hash_token : hash_tokens){
      std::cout << hash_token << " <><> " << std::endl;
    }
    std::cout << "Printing Hash tokens Done!" << std::endl;
    return hash_tokens;
  }

  std::vector<std::string> toStrings(const std::string& input) final {
    return text::charKGrams(input, _k);
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

  ar::ConstArchivePtr toArchive() const final {
    auto map = ar::Map::make();
    map->set("type", ar::str(type()));
    map->set("k", ar::u64(_k));
    return map;
  }

  static std::string type() { return "char_k_gram"; }

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