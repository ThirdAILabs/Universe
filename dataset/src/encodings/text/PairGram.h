#pragma once

#include "TextEncodingInterface.h"
#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <algorithm>
#include <cctype>
#include <ctype.h>
#include <limits>
#include <sstream>

namespace thirdai::dataset {

/**
 * Encodes a sentence as a weighted set of ordered pairs of words.
 */
struct PairGram : public TextEncoding {
  explicit PairGram(uint32_t dim = 100000) : _dim(dim) {}

  void encodeText(const std::string& text, ExtendableVector& vec) final {
    
    // TODO(Geordie): Do we need to make lower case?
    std::string lower_case_text = text;
    for (auto& c : lower_case_text) {
      c = std::tolower(c);
    }
    
    std::vector<uint32_t> pairgram_indices;
    std::vector<uint32_t> seen_unigram_hashes;
    
    // Add pairgrams as we encounter new words.
    const auto* start_ptr = lower_case_text.c_str();
    bool last_ptr_was_space = false;
    for (const auto& c : lower_case_text) {

      // Just saw a word boundary
      if (isspace(c) && !last_ptr_was_space) {
        last_ptr_was_space = true;
        addPairGrams(seen_unigram_hashes, start_ptr, std::distance(start_ptr, &c), pairgram_indices);
      }

      // Encountered a new word
      if (last_ptr_was_space && !isspace(c)) {
        last_ptr_was_space = false;
        start_ptr = &c;
      }
    }

    // Don't leave out last word.
    if (!last_ptr_was_space) {
      size_t cur_pos = std::distance(text.c_str(), start_ptr);
      addPairGrams(seen_unigram_hashes, start_ptr, text.size() - cur_pos, pairgram_indices);
    }

    vec.incrementExtensionAtIndices(pairgram_indices, 1.0);
  }

  uint32_t featureDim() final { return _dim; }

  bool isDense() final { return false; }

 private:

  inline void addPairGrams(std::vector<uint32_t>& prev_unigram_hashes, const char* start_ptr, size_t len, std::vector<uint32_t>& pair_gram_indices) const {
    // Hash the new word
    uint32_t new_unigram_hash =
            hashing::MurmurHash(start_ptr, len, /* seed = */ 341) % _dim;

    // Add new unigram here because same-word pairgrams also help.
    prev_unigram_hashes.push_back(new_unigram_hash);

    // Create ordered pairgrams by pairing with all previous words (including this one).
    // Combine the hashes of the unigrams that make up the pairgram.
    for (const auto& prev_word_hash : prev_unigram_hashes) {
      pair_gram_indices.push_back(
          hashing::HashUtils::combineHashes(prev_word_hash, new_unigram_hash));
    }
  }

  uint32_t _dim;
};

}  // namespace thirdai::dataset