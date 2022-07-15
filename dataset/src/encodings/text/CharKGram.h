#pragma once

#include "TextEncodingInterface.h"
#include "TextEncodingUtils.h"
#include <hashing/src/MurmurHash.h>
#include <dataset/src/blocks/BlockInterface.h>

namespace thirdai::dataset {

/**
 * Encodes a sentence as a weighted set of character k-grams.
 */
class CharKGram : public TextEncoding {
 public:
  /**
   * Constructor. Accepts the number of characters in each k-gram and
   * the desired dimension of the encoding.
   */
  explicit CharKGram(
      uint32_t k, uint32_t dim = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM)
      : _k(k), _dim(dim) {}

  void encodeText(const std::string_view text,
                  SegmentedFeatureVector& vec) final {
    // TODO(Geordie): Do we need to make lower case?
    std::string lower_case_text = TextEncodingUtils::makeLowerCase(text);

    std::vector<uint32_t> char_k_grams;

    // TODO(Geordie): Do we need to pad start and end of text by k-1 characters?
    for (uint32_t offset = 0; offset < text.size() - (_k - 1); offset++) {
      uint32_t k_gram_hash =
          TextEncodingUtils::computeUnigram(&lower_case_text.at(offset), _k) %
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
  }

  uint32_t featureDim() final { return _dim; }

  bool isDense() final { return false; }

 private:
  uint32_t _k;
  uint32_t _dim;
};

}  // namespace thirdai::dataset