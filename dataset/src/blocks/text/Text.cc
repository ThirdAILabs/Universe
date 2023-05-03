#include "Text.h"

namespace thirdai::dataset {

Explanation TextBlock::explainIndex(uint32_t index_within_block,
                                    ColumnarInputSample& input) {
  std::string_view string = input.column(_col);

  // assign outside the if statement so string view below wont point to an out
  // of scope string variable
  std::string lowercase_string;
  if (_lowercase) {
    lowercase_string = text::lower(string);
    string = lowercase_string;
  }
  std::vector<std::string_view> tokens = _tokenizer->apply(string);
  std::string keyword =
      _encoder->getResponsibleWord(tokens, index_within_block, _dim);

  return {_col, keyword};
}

void TextBlock::buildSegment(ColumnarInputSample& input,
                             SegmentedFeatureVector& vec) {
  std::string_view string = input.column(_col);

  // assign outside the if statement so string view below wont point to an out
  // of scope string variable
  std::string lowercase_string;
  if (_lowercase) {
    lowercase_string = text::lower(string);
    string = lowercase_string;
  }

  std::vector<std::string_view> tokens = _tokenizer->apply(string);
  std::vector<uint32_t> indices = _encoder->apply(tokens);
  token_encoding::mod(indices, _dim);

  for (auto& [index, value] : token_encoding::sumRepeatedIndices(indices)) {
    vec.addSparseFeatureToSegment(index, value);
  }
}

}  // namespace thirdai::dataset