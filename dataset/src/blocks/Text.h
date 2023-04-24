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

class TextTokenizer {
 public:
  virtual std::vector<std::string_view> apply(
      const std::string_view& input) = 0;

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

  std::vector<std::string_view> apply(const std::string_view& input) final {
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

  std::vector<std::string_view> apply(const std::string_view& input) final {
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

  std::vector<std::string_view> apply(const std::string_view& input) final {
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

class TextEncoder {
 public:
  virtual std::vector<uint32_t> apply(
      const std::vector<std::string_view>& tokens) = 0;

  virtual std::string getResponsibleWord(
      const std::vector<std::string_view>& tokens,
      uint32_t index_within_block) {
    (void)tokens;
    (void)index_within_block;
    throw std::invalid_argument(
        "Explanations are not supported for this type of encoding. ");
  }

  virtual ~TextEncoder() = default;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using TextEncoderPtr = std::shared_ptr<TextEncoder>;

class NGramEncoder : public TextEncoder {
 public:
  explicit NGramEncoder(uint32_t n) : _n(n) {}

  static auto make(uint32_t n) { return std::make_shared<NGramEncoder>(n); }

  std::vector<uint32_t> apply(
      const std::vector<std::string_view>& tokens) final {
    return token_encoding::ngrams(token_encoding::hashTokens(tokens), _n);
  }

 private:
  uint32_t _n;

  NGramEncoder() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextEncoder>(this), _n);
  }
};

class PairGramEncoder : public TextEncoder {
 public:
  PairGramEncoder() {}

  static auto make() { return std::make_shared<PairGramEncoder>(); }

  std::vector<uint32_t> apply(
      const std::vector<std::string_view>& tokens) final {
    return token_encoding::pairgrams(token_encoding::hashTokens(tokens));
  }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextEncoder>(this));
  }
};

/**
 * A block that encodes text (e.g. sentences / paragraphs).
 */
class TextBlock : public Block {
 public:
  explicit TextBlock(ColumnIdentifier col, TextTokenizerPtr tokenizer,
                     TextEncoderPtr encoder,
                     uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM)
      : _dim(dim),
        _col(std::move(col)),
        _tokenizer(std::move(tokenizer)),
        _encoder(std::move(encoder)) {}

  static auto make(ColumnIdentifier col, TextTokenizerPtr tokenizer,
                   TextEncoderPtr encoder,
                   uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<TextBlock>(col, tokenizer, encoder, dim);
  }

  static auto make(ColumnIdentifier col, TextTokenizerPtr tokenizer,
                   uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<TextBlock>(
        col, tokenizer, dataset::NGramEncoder::make(/* n = */ 1), dim);
  }

  static auto make(ColumnIdentifier col, TextEncoderPtr encoder,
                   uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<TextBlock>(col, NaiveSplitTokenizer::make(),
                                       encoder, dim);
  }

  static auto make(ColumnIdentifier col,
                   uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<TextBlock>(col, NaiveSplitTokenizer::make(),
                                       NGramEncoder::make(/* n = */ 1), dim);
  }

  uint32_t featureDim() const final { return _dim; };

  bool isDense() const final { return false; };

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final {
    std::string_view text = input.column(_col);
    std::vector<std::string_view> tokens = _tokenizer->apply(text);
    std::string keyword =
        _encoder->getResponsibleWord(tokens, index_within_block);

    return {_col, keyword};
  }

 protected:
  void buildSegment(ColumnarInputSample& input,
                    SegmentedFeatureVector& vec) final {
    std::string_view text = input.column(_col);

    std::vector<std::string_view> tokens = _tokenizer->apply(text);
    std::vector<uint32_t> indices = _encoder->apply(tokens);
    token_encoding::mod(indices, _dim);

    for (auto& [index, value] : token_encoding::sumRepeatedIndices(indices)) {
      vec.addSparseFeatureToSegment(index, value);
    }
  }

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final {
    return {&_col};
  };

 private:
  // Constructor for cereal.
  TextBlock() {}

  uint32_t _dim;
  ColumnIdentifier _col;
  TextTokenizerPtr _tokenizer;
  TextEncoderPtr _encoder;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _dim, _col, _tokenizer, _encoder);
  }
};

using TextBlockPtr = std::shared_ptr<TextBlock>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TextBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::NaiveSplitTokenizer)
CEREAL_REGISTER_TYPE(thirdai::dataset::WordPunctTokenizer)
CEREAL_REGISTER_TYPE(thirdai::dataset::CharKGramTokenizer)
CEREAL_REGISTER_TYPE(thirdai::dataset::PairGramEncoder)
CEREAL_REGISTER_TYPE(thirdai::dataset::NGramEncoder)