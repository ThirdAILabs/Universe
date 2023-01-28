#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <dataset/src/utils/TokenEncoding.h>
#include <utils/StringManipulation.h>
#include <memory>
#include <stdexcept>

namespace thirdai::dataset {

/**
 * A block that encodes text (e.g. sentences / paragraphs).
 */
class TextBlock : public Block {
 public:
  explicit TextBlock(ColumnIdentifier col, uint32_t dim)
      : _dim(dim), _col(std::move(col)) {}

  uint32_t featureDim() const final { return _dim; };

  bool isDense() const final { return false; };

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final {
    return {_col, getResponsibleWord(index_within_block, input.column(_col))};
  }

  virtual std::string getResponsibleWord(
      uint32_t index, const std::string_view& text) const = 0;

 protected:
  std::exception_ptr buildSegment(ColumnarInputSample& input,
                                  SegmentedFeatureVector& vec) final {
    return encodeText(input.column(_col), vec);
  }

  virtual std::exception_ptr encodeText(std::string_view text,
                                        SegmentedFeatureVector& vec) = 0;

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final {
    return {&_col};
  };

  uint32_t _dim;

  // Constructor for cereal.
  TextBlock() {}

 private:
  ColumnIdentifier _col;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _dim, _col);
  }
};

using TextBlockPtr = std::shared_ptr<TextBlock>;

/**
 * A block that encodes text as a weighted set of ordered pairs of
 * space-separated words.
 */
class PairGramTextBlock final : public TextBlock {
 public:
  explicit PairGramTextBlock(
      ColumnIdentifier col,
      uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM)
      : TextBlock(std::move(col), dim) {}

  static auto make(ColumnIdentifier col,
                   uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<PairGramTextBlock>(std::move(col), dim);
  }

  std::string getResponsibleWord(uint32_t index,
                                 const std::string_view& text) const final {
    (void)index;
    (void)text;
    throw std::invalid_argument(
        "Explain Index is not yet implemented for pairgram block.");
  }

 protected:
  std::exception_ptr encodeText(std::string_view text,
                                SegmentedFeatureVector& vec) final {
    std::vector<uint32_t> pairgrams =
        token_encoding::pairgrams(token_encoding::tokenize(text::split(text)));
    token_encoding::mod(pairgrams, _dim);
    for (auto& [index, value] : token_encoding::sumRepeatedIndices(pairgrams)) {
      vec.addSparseFeatureToSegment(index, value);
    }

    return nullptr;
  }

 private:
  // Private constructor for cereal.
  PairGramTextBlock() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextBlock>(this));
  }
};

using PairGramTextBlockPtr = std::shared_ptr<PairGramTextBlock>;

/**
 * A block that encodes text as hashed N-gram tokens.
 */
class NGramTextBlock final : public TextBlock {
 public:
  explicit NGramTextBlock(
      ColumnIdentifier col, uint32_t n = 1,
      uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM,
      char delimiter = ' ')
      : TextBlock(std::move(col), dim), _n(n), _delimiter(delimiter) {}

  static auto make(ColumnIdentifier col, uint32_t n = 1,
                   uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM,
                   char delimiter = ' ') {
    return std::make_shared<NGramTextBlock>(std::move(col), n, dim, delimiter);
  }

  std::string getResponsibleWord(uint32_t index,
                                 const std::string_view& text) const final {
    // TODO(any): implement explanations for generic N grams
    if (_n != 1) {
      throw std::invalid_argument(
          "Word explanations not supported for this type of featurization.");
    }
    std::unordered_map<uint32_t, std::string> index_to_word_map =
        token_encoding::buildUnigramHashToWordMap(text, _dim, _delimiter);
    return index_to_word_map.at(index);
  }

 protected:
  std::exception_ptr encodeText(std::string_view text,
                                SegmentedFeatureVector& vec) final {
    std::vector<uint32_t> ngrams =
        token_encoding::ngrams(text, /* n= */ _n, _delimiter);
    token_encoding::mod(ngrams, _dim);

    for (auto& [index, value] : token_encoding::sumRepeatedIndices(ngrams)) {
      vec.addSparseFeatureToSegment(index, value);
    }

    return nullptr;
  }

 private:
  // Private constructor for cereal.
  NGramTextBlock() {}

  uint32_t _n;
  char _delimiter = ' ';

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextBlock>(this), _n, _delimiter);
  }
};

using NGramTextBlockPtr = std::shared_ptr<NGramTextBlock>;

/**
 * A block that encodes text as a weighted set of character k-grams.
 */
class CharKGramTextBlock final : public TextBlock {
 public:
  CharKGramTextBlock(ColumnIdentifier col, uint32_t k,
                     uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM)
      : TextBlock(std::move(col), dim), _k(k) {}

  static auto make(ColumnIdentifier col, uint32_t k,
                   uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<CharKGramTextBlock>(std::move(col), k, dim);
  }

  std::string getResponsibleWord(uint32_t index,
                                 const std::string_view& text) const final {
    (void)index;
    (void)text;
    throw std::invalid_argument(
        "Explain Index is not yet implemented for char-k block.");
  }

 protected:
  std::exception_ptr encodeText(std::string_view text,
                                SegmentedFeatureVector& vec) final {
    if (text.empty()) {
      return nullptr;
    }
    std::string lower_case_text = text::lower(text);

    std::vector<uint32_t> char_k_grams;

    size_t n_kgrams = text.size() >= _k ? text.size() - (_k - 1) : 1;
    size_t len = std::min(text.size(), static_cast<size_t>(_k));
    for (uint32_t offset = 0; offset < n_kgrams; offset++) {
      uint32_t k_gram_hash = token_encoding::seededMurmurHash(
                                 /* key= */ &lower_case_text.at(offset), len) %
                             _dim;
      char_k_grams.push_back(k_gram_hash);
    }

    /*
      Deduplication adds an overhead of around 10% but helps to reduce
      number of entries in the sparse vector, which can in turn make BOLT
      run faster.
    */
    for (auto& [index, value] :
         token_encoding::sumRepeatedIndices(char_k_grams)) {
      vec.addSparseFeatureToSegment(index, value);
    }

    return nullptr;
  }

 private:
  uint32_t _k;

  // Private constructor for cereal.
  CharKGramTextBlock() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextBlock>(this), _k);
  }
};

using CharKGramTextBlockPtr = std::shared_ptr<CharKGramTextBlock>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TextBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::PairGramTextBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::NGramTextBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::CharKGramTextBlock)