#include "Text.h"
#include <utils/text/RegexPatterns.h>

namespace thirdai::dataset {

Explanation TextBlock::explainIndex(uint32_t index_within_block,
                                    ColumnarInputSample& input) {
  std::string string = input.column(_col);

  if (_lowercase) {
    string = text::lower(string);
  }

  std::vector<uint32_t> tokens = _tokenizer->tokenize(string);

  uint32_t source_token =
      _encoder->undoEncoding(tokens, index_within_block, _dim);

  std::string keyword = _tokenizer->getResponsibleWord(string, source_token);

  return {_col, keyword};
}

void TextBlock::buildSegment(ColumnarInputSample& input,
                             SegmentedFeatureVector& vec) {
  std::string string = input.column(_col);

  if (_lowercase) {
    string = text::lower(string);
  }

  std::vector<uint32_t> tokens = _tokenizer->tokenize(string);
  std::vector<uint32_t> indices = _encoder->encode(tokens);
  token_encoding::mod(indices, _dim);

  for (auto& [index, value] : token_encoding::sumRepeatedIndices(indices)) {
    vec.addSparseFeatureToSegment(index, value);
  }
}

}  // namespace thirdai::dataset