#pragma once

#include "TextEncodingInterface.h"
#include "TextEncodingUtils.h"
#include <hashing/src/HashUtils.h>
#include <dataset/src/blocks/BlockInterface.h>

namespace thirdai::dataset {

/**
 * Encodes a sentence as a weighted set of ordered pairs of words.
 */
class PairGram : public TextEncoding {
 public:
  /**
   * Constructor. Accepts the desired dimension of the encoding.
   */
  explicit PairGram(uint32_t dim = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM)
      : _dim(dim) {}

  void encodeText(const std::string_view text,
                  SegmentedFeatureVector& vec) final {
    TextEncodingUtils::computePairgrams(text, _dim, vec);
  }

  uint32_t featureDim() final { return _dim; }

  bool isDense() final { return false; }

 private:
  uint32_t _dim;
};

}  // namespace thirdai::dataset