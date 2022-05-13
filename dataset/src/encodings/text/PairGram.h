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
 * Encodes a sentence as a weighted set of ordered pairs of words.
 */
struct PairGram : public TextEncoding {
  /**
   * Constructor. Accepts the desired dimension of the encoding.
   */
  explicit PairGram(uint32_t dim = 100000) : _dim(dim) {}

  void encodeText(const std::string& text, ExtendableVector& vec) final {
    // TODO(Geordie): Do we need to make lower case?
    std::string lower_case_text = text;
    for (auto& c : lower_case_text) {
      c = std::tolower(c);
    }

    std::vector<uint32_t> seen_unigram_hashes;
    std::vector<uint32_t> pair_grams;

    TextEncodingUtils::forEachWord(
        lower_case_text, [&](const char* start_ptr, size_t len) {
          addPairGrams(seen_unigram_hashes, start_ptr, len, pair_grams);
        });

    // Deduplication helps to reduce number of entries in the sparse
    // vector but has huge overheads. May want to remove in a future iteration.
    TextEncodingUtils::sumRepeatedIndices(pair_grams, 1.0, vec);
  }

  uint32_t featureDim() final { return _dim; }

  bool isDense() final { return false; }

 private:
  inline void addPairGrams(std::vector<uint32_t>& prev_unigram_hashes,
                           const char* start_ptr, size_t len,
                           std::vector<uint32_t>& pair_grams) const {
    // Hash the new word
    uint32_t new_unigram_hash =
        hashing::MurmurHash(start_ptr, len, /* seed = */ 341);

    // Add new unigram here because same-word pairgrams also help.
    prev_unigram_hashes.push_back(new_unigram_hash);

    // Create ordered pairgrams by pairing with all previous words (including
    // this one). Combine the hashes of the unigrams that make up the pairgram.
    for (const auto& prev_word_hash : prev_unigram_hashes) {
      uint32_t pair_gram =
          hashing::HashUtils::combineHashes(prev_word_hash, new_unigram_hash) %
          _dim;
      pair_grams.push_back(pair_gram);
    }
  }

  uint32_t _dim;
};

}  // namespace thirdai::dataset