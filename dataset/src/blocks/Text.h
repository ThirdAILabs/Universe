#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <dataset/src/utils/TextEncodingUtils.h>
#include <utils/StringManipulation.h>
#include <memory>
#include <stdexcept>

namespace thirdai::dataset {

/**
 * A block that encodes text (e.g. sentences / paragraphs).
 */
class TextBlock : public Block {
 public:
  explicit TextBlock(uint32_t col, uint32_t dim) : _dim(dim), _col(col) {}

  uint32_t featureDim() const final { return _dim; };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final { return _col + 1; };

  Explanation explainIndex(
      uint32_t index_within_block,
      const std::vector<std::string_view>& input_row) final {
    return {_col, getResponsibleWord(index_within_block, input_row.at(_col))};
  }

  virtual std::string getResponsibleWord(
      uint32_t index, const std::string_view& text) const = 0;

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    return encodeText(input_row.at(_col), vec);
  }

  virtual std::exception_ptr encodeText(std::string_view text,
                                        SegmentedFeatureVector& vec) = 0;

  uint32_t _dim;

  // Constructor for cereal.
  TextBlock() {}

 private:
  uint32_t _col;

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
      uint32_t col, uint32_t dim = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM)
      : TextBlock(col, dim) {}

  static auto make(
      uint32_t col,
      uint32_t dim = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<PairGramTextBlock>(col, dim);
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
        TextEncodingUtils::computeRawPairgrams(text, _dim);

    TextEncodingUtils::sumRepeatedIndices(
        pairgrams, /* base_value= */ 1.0, [&](uint32_t pairgram, float value) {
          vec.addSparseFeatureToSegment(pairgram, value);
        });

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
 * A block that encodes text as a weighted set of space-separated words.
 */
class UniGramTextBlock final : public TextBlock {
 public:
  explicit UniGramTextBlock(
      uint32_t col, uint32_t dim = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM)
      : TextBlock(col, dim) {}

  static auto make(
      uint32_t col,
      uint32_t dim = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<UniGramTextBlock>(col, dim);
  }

  std::string getResponsibleWord(uint32_t index,
                                 const std::string_view& text) const final {
    std::unordered_map<uint32_t, std::string> index_to_word_map =
        TextEncodingUtils::buildUnigramHashToWordMap(text, _dim);
    return index_to_word_map.at(index);
  }

 protected:
  std::exception_ptr encodeText(std::string_view text,
                                SegmentedFeatureVector& vec) final {
    std::vector<uint32_t> unigrams =
        TextEncodingUtils::computeRawUnigramsWithRange(text, _dim);

    TextEncodingUtils::sumRepeatedIndices(
        unigrams, /* base_value= */ 1.0, [&](uint32_t unigram, float value) {
          vec.addSparseFeatureToSegment(unigram, value);
        });

    return nullptr;
  }

 private:
  // Private constructor for cereal.
  UniGramTextBlock() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextBlock>(this));
  }
};

using UniGramTextBlockPtr = std::shared_ptr<UniGramTextBlock>;

/**
 * A block that encodes text as a weighted set of character k-grams.
 */
class CharKGramTextBlock final : public TextBlock {
 public:
  CharKGramTextBlock(
      uint32_t col, uint32_t k,
      uint32_t dim = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM)
      : TextBlock(col, dim), _k(k) {}

  static auto make(
      uint32_t col, uint32_t k,
      uint32_t dim = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<CharKGramTextBlock>(col, k, dim);
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
    std::string lower_case_text = utils::lower(text);

    std::vector<uint32_t> char_k_grams;

    size_t n_kgrams = text.size() >= _k ? text.size() - (_k - 1) : 1;
    size_t len = std::min(text.size(), static_cast<size_t>(_k));
    for (uint32_t offset = 0; offset < n_kgrams; offset++) {
      uint32_t k_gram_hash = TextEncodingUtils::computeUnigram(
                                 /* key= */ &lower_case_text.at(offset), len) %
                             _dim;
      char_k_grams.push_back(k_gram_hash);
    }

    /*
      Deduplication adds an overhead of around 10% but helps to reduce
      number of entries in the sparse vector, which can in turn make BOLT
      run faster.
    */
    TextEncodingUtils::sumRepeatedIndices(
        /* indices = */ char_k_grams,
        /* base_value = */ 1.0, [&](uint32_t index, float value) {
          vec.addSparseFeatureToSegment(index, value);
        });

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
CEREAL_REGISTER_TYPE(thirdai::dataset::UniGramTextBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::CharKGramTextBlock)