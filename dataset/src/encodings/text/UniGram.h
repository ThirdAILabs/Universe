#pragma once

#include "TextEncodingInterface.h"
#include "TextEncodingUtils.h"
#include <hashing/src/MurmurHash.h>
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

  void encodeText(const std::string& text, ExtendableVector& vec) final {
    // TODO(Geordie): Do we need to make lower case?
    std::string lower_case_text = TextEncodingUtils::makeLowerCase(text);

    std::vector<uint32_t> uni_grams;

    TextEncodingUtils::forEachWordHash(
        lower_case_text,
        [&](uint32_t word_hash) { uni_grams.push_back(word_hash % _dim); });

    // Deduplication adds an overhead of around 10% but helps to reduce
    // number of entries in the sparse vector, which can in turn make BOLT
    // run faster.
    TextEncodingUtils::sumRepeatedIndices(uni_grams, 1.0, vec);
  }

  uint32_t featureDim() final { return _dim; }

  bool isDense() final { return false; }

 private:
  uint32_t _dim;
};

}  // namespace thirdai::dataset