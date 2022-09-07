#pragma once

#include "BlockInterface.h"
#include <dataset/src/utils/TextEncodingUtils.h>
#include <memory>

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

  uint32_t getColumnNum() const final { return _col; }

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    return encodeText(input_row.at(_col), vec);
  }

  virtual std::exception_ptr encodeText(std::string_view text,
                                        SegmentedFeatureVector& vec) = 0;

  uint32_t _dim;

 private:
  uint32_t _col;
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

  std::string at(uint32_t index) {
    return index_to_word_map.at(index);
  }


 protected:
  std::exception_ptr encodeText(std::string_view text,
                                SegmentedFeatureVector& vec) final {
    auto [unigrams, index_to_word] =
        TextEncodingUtils::computeRawUnigramsWithRange(text, _dim,true);

    index_to_word_map = std::move(*index_to_word);

    TextEncodingUtils::sumRepeatedIndices(
        unigrams, /* base_value= */ 1.0, [&](uint32_t unigram, float value) {
          vec.addSparseFeatureToSegment(unigram, value);
        });

    return nullptr;
  }

  std::unordered_map<uint32_t,std::string> index_to_word_map;

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

 protected:
  std::exception_ptr encodeText(std::string_view text,
                                SegmentedFeatureVector& vec) final {
    std::string lower_case_text = TextEncodingUtils::makeLowerCase(text);

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
};

using CharKGramTextBlockPtr = std::shared_ptr<CharKGramTextBlock>;

}  // namespace thirdai::dataset