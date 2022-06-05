#pragma once

#include "TextEncodingInterface.h"
#include "TextEncodingUtils.h"
#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>

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

    TextEncodingUtils::forEachWord(lower_case_text, [&](std::string_view word) {
      uint32_t hash = hashing::MurmurHash(word.cbegin(), word.size(),
                                          TextEncodingUtils::HASH_SEED) %
                      _dim;
      uni_grams.push_back(hash);
    });

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