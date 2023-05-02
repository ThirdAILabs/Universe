#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <dataset/src/Vocabulary.h>
#include <utils/StringManipulation.h>
#include <string>

namespace thirdai::dataset {

class TextTokenizer {
 public:
  virtual std::vector<std::string> apply(const std::string& input) = 0;

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

  std::vector<std::string> apply(const std::string& input) final {
    return text::split(input, _delimiter);
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

  std::vector<std::string> apply(const std::string& input) final {
    return text::tokenizeSentence(input);
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

  std::vector<std::string> apply(const std::string& input) final {
    return text::charKGrams(input, _k);
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

// class WordpieceTokenizer : public TextTokenizer {
//  public:
//   explicit WordpieceTokenizer(std::shared_ptr<WordpieceVocab> vocab)
//       : _vocab(std::move(vocab)) {}

//   static auto make(uint32_t k) {
//     return std::make_shared<CharKGramTokenizer>(k);
//   }

//   std::vector<std::string_view> apply(const std::string_view& input) final {
//     return _vocab->tokenize(input);
//   }

//  private:
//   std::shared_ptr<WordpieceVocab> _vocab;

//   WordpieceTokenizer() {}

//   friend class cereal::access;
//   template <class Archive>
//   void serialize(Archive& archive) {
//     archive(cereal::base_class<TextTokenizer>(this), _vocab);
//   }
// };

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::NaiveSplitTokenizer)
CEREAL_REGISTER_TYPE(thirdai::dataset::WordPunctTokenizer)
CEREAL_REGISTER_TYPE(thirdai::dataset::CharKGramTokenizer)