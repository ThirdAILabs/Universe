#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "TextEncoder.h"
#include "TextTokenizer.h"
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
                     TextEncoderPtr encoder, bool lowercase = false,
                     uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM)
      : _col(std::move(col)),
        _lowercase(lowercase),
        _tokenizer(std::move(tokenizer)),
        _encoder(std::move(encoder)),
        _dim(dim) {}

  static auto make(const ColumnIdentifier& col,
                   const TextTokenizerPtr& tokenizer,
                   const TextEncoderPtr& encoder, bool lowercase = false,
                   uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<TextBlock>(col, tokenizer, encoder, lowercase, dim);
  }

  static auto make(const ColumnIdentifier& col,
                   const TextTokenizerPtr& tokenizer, bool lowercase = false,
                   uint32_t dim = token_encoding::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<TextBlock>(col, tokenizer,
                                       dataset::NGramEncoder::make(/* n = */ 1),
                                       lowercase, dim);
  }

  uint32_t featureDim() const final { return _dim; };

  bool isDense() const final { return false; };

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final;

  bool lowercase() const { return _lowercase; }

  TextTokenizerPtr tokenizer() const { return _tokenizer; }

  TextEncoderPtr encoder() const { return _encoder; }

 protected:
  void buildSegment(ColumnarInputSample& input,
                    SegmentedFeatureVector& vec) final;

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final {
    return {&_col};
  };

 private:
  // Constructor for cereal.
  TextBlock() {}

  ColumnIdentifier _col;
  bool _lowercase;
  TextTokenizerPtr _tokenizer;
  TextEncoderPtr _encoder;
  uint32_t _dim;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _col, _lowercase, _tokenizer,
            _encoder, _dim);
  }
};

using TextBlockPtr = std::shared_ptr<TextBlock>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TextBlock)