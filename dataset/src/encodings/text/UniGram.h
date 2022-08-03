#pragma once

#include "TextEncodingInterface.h"
#include "TextEncodingUtils.h"
#include <dataset/src/blocks/BlockInterface.h>

namespace thirdai::dataset {

/**
 * Encodes a sentence as a weighted set of words.
 */
class UniGram : public TextEncoding {
 public:
  /**
   * Constructor. Accepts the desired dimension of the encoding.
   */
  explicit UniGram(uint32_t dim = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM)
      : _dim(dim) {}

  void encodeText(const std::string_view text,
                  SegmentedFeatureVector& vec) final {
    std::vector<uint32_t> unigrams =
        TextEncodingUtils::computeRawUnigramsWithRange(text, _dim);

    TextEncodingUtils::sumRepeatedIndices(
        unigrams, /* base_value= */ 1.0, [&](uint32_t unigram, float value) {
          vec.addSparseFeatureToSegment(unigram, value);
        });
  }

  uint32_t featureDim() final { return _dim; }

  bool isDense() final { return false; }

 private:
  uint32_t _dim;
};

}  // namespace thirdai::dataset