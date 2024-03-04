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
  explicit NaiveSplitTokenizer(char delimiter = ' ', uint32_t seed = 341)
      : _delimiter(delimiter), _seed(seed) {}

  static auto make(char delimiter = ' ', uint32_t seed = 341) {
    return std::make_shared<NaiveSplitTokenizer>(delimiter, seed);
  }

  std::vector<uint32_t> tokenize(const std::string& input) final {
    return token_encoding::hashTokens(text::split(input, _delimiter), _seed);
  }

  std::string getResponsibleWord(const std::string& input,
                                 uint32_t source_token) final {
    auto map = token_encoding::buildUnigramHashToWordMap(
        text::split(input, _delimiter), _seed);

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
  uint32_t _seed;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextTokenizer>(this), _delimiter, _seed);
  }
};

class WordPunctTokenizer : public TextTokenizer {
 public:
  explicit WordPunctTokenizer(uint32_t seed = 341) : _seed(seed) {}

  static auto make(uint32_t seed = 341) {
    return std::make_shared<WordPunctTokenizer>(seed);
  }

  std::vector<uint32_t> tokenize(const std::string& input) final {
    return token_encoding::hashTokens(text::tokenizeSentence(input), _seed);
  }

  std::string getResponsibleWord(const std::string& input,
                                 uint32_t source_token) final {
    auto map = token_encoding::buildUnigramHashToWordMap(
        text::tokenizeSentence(input), _seed);

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
  uint32_t _seed;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextTokenizer>(this), _seed);
  }
};

class CharKGramTokenizer : public TextTokenizer {
 public:
  explicit CharKGramTokenizer(uint32_t k, uint32_t seed = 341)
      : _k(k), _seed(seed) {}

  static auto make(uint32_t k, uint32_t seed = 341) {
    return std::make_shared<CharKGramTokenizer>(k, seed);
  }

  std::vector<uint32_t> tokenize(const std::string& input) final {
    return token_encoding::hashTokens(text::charKGrams(input, _k), _seed);
  }

  std::string getResponsibleWord(const std::string& input,
                                 uint32_t source_token) final {
    auto map = token_encoding::buildUnigramHashToWordMap(
        text::charKGrams(input, _k), _seed);

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
  uint32_t _seed;

  CharKGramTokenizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextTokenizer>(this), _k, _seed);
  }
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::NaiveSplitTokenizer)
CEREAL_REGISTER_TYPE(thirdai::dataset::WordPunctTokenizer)
CEREAL_REGISTER_TYPE(thirdai::dataset::CharKGramTokenizer)