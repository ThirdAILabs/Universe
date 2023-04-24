#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "TextEncoders.h"
#include "TextTokenizers.h"
#include <dataset/src/blocks/BlockInterface.h>
#include <memory>
#include <stdexcept>

namespace thirdai::dataset {

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
        _encoder->getResponsibleWord(tokens, index_within_block, _dim);

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