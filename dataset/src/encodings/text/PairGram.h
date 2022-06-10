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

  void encodeText(const std::string_view text, SegmentedFeatureVector& vec) final {
    // TODO(Geordie): Do we need to make lower case?
    std::string lower_case_text = TextEncodingUtils::makeLowerCase(text);

    std::vector<uint32_t> seen_unigram_hashes;
    std::vector<uint32_t> pair_grams;

    TextEncodingUtils::forEachWordHash(
        lower_case_text, [&](uint32_t word_hash) {
          addPairGrams(seen_unigram_hashes, word_hash, pair_grams);
        });

    // Deduplication helps to reduce number of entries in the sparse
    // vector but has huge overheads. May want to remove in a future iteration.
    // We do this instead of using a map because at this scale, sorting and
    // deduplicating is still faster than map's O(1) insertions. Additionally,
    // iterating over a map is slow
    TextEncodingUtils::sumRepeatedIndices(pair_grams, 1.0, vec);
  }

  uint32_t featureDim() final { return _dim; }

  bool isDense() final { return false; }

 private:
  inline void addPairGrams(std::vector<uint32_t>& prev_unigram_hashes,
                           uint32_t word_hash,
                           std::vector<uint32_t>& pair_grams) const {
    // Add new unigram here because same-word pairgrams also help.
    prev_unigram_hashes.push_back(word_hash);

    // Create ordered pairgrams by pairing with all previous words (including
    // this one). Combine the hashes of the unigrams that make up the pairgram.
    for (const auto& prev_word_hash : prev_unigram_hashes) {
      uint32_t pair_gram =
          hashing::HashUtils::combineHashes(prev_word_hash, word_hash) % _dim;
      pair_grams.push_back(pair_gram);
    }
  }

  uint32_t _dim;
};

}  // namespace thirdai::dataset