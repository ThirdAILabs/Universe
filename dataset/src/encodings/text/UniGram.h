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
  // explicit UniGram(uint32_t dim = 100000) 
  //                  : _dim(dim), _start_pos(0), _end_pos(std::numeric_limits<uint32_t>::max()) {}

  explicit UniGram(uint32_t dim = 100000, 
                   uint32_t start_pos=0,
                   uint32_t end_pos=std::numeric_limits<uint32_t>::max()) 
                   : _dim(dim), _start_pos(start_pos), _end_pos(end_pos) {}

  void encodeText(const std::string& text, ExtendableVector& vec) final {
    // TODO(Geordie): Do we need to make lower case?
    std::string lower_case_text = TextEncodingUtils::makeLowerCase(text);

    std::vector<uint32_t> uni_grams;

    TextEncodingUtils::forEachWord(
        lower_case_text, [&](const char* start_ptr, size_t len) {
          uint32_t hash =
              hashing::MurmurHash(start_ptr, len, /* seed = */ 341) % _dim;
          uni_grams.push_back(hash);
        }, _start_pos, _end_pos);

    // Deduplication adds an overhead of around 10% but helps to reduce
    // number of entries in the sparse vector, which can in turn make BOLT
    // run faster.
    TextEncodingUtils::sumRepeatedIndices(uni_grams, 1.0, vec);
  }

  uint32_t featureDim() final { return _dim; }

  bool isDense() final { return false; }

 private:
  uint32_t _dim;
  uint32_t _start_pos;
  uint32_t _end_pos;
};

}  // namespace thirdai::dataset