#pragma once

#include <hashing/src/MurmurHash.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <functional>
#include <string_view>
#include <type_traits>

namespace thirdai::dataset {

class TextEncodingUtils {
 public:
  static constexpr uint32_t HASH_SEED = 341;
  static constexpr uint32_t DEFAULT_TEXT_ENCODING_DIM = 100000;

  /**
   * Parses through a sentence and applied a function to each word.
   */
  template <typename WORD_PROCESSOR_T>
  inline static void forEachWordHash(const std::string_view sentence,
                                     WORD_PROCESSOR_T word_processor) {
    static_assert(std::is_convertible<WORD_PROCESSOR_T,
                                      std::function<void(uint32_t)>>::value);

    const auto* start_ptr = sentence.c_str();
    bool last_ptr_was_space = false;
    for (const auto& c : sentence) {
      // Just saw a word boundary
      if (isspace(c) && !last_ptr_was_space) {
        last_ptr_was_space = true;
        size_t len = std::distance(start_ptr, &c);
        uint32_t word_hash = hashing::MurmurHash(start_ptr, len, HASH_SEED);
        word_processor(word_hash);
      }

      // Encountered a new word
      if (last_ptr_was_space && !isspace(c)) {
        last_ptr_was_space = false;
        start_ptr = &c;
      }
    }

    // Don't leave out last word.
    if (!last_ptr_was_space) {
      size_t cur_pos = std::distance(sentence.c_str(), start_ptr);
      uint32_t len = sentence.size() - cur_pos;
      uint32_t word_hash = hashing::MurmurHash(start_ptr, len, HASH_SEED);
      word_processor(word_hash);
    }
  }

  static std::vector<uint32_t> computeUnigrams(const std::string_view sentence,
                                               uint32_t output_range) {
    std::vector<uint32_t> unigrams;
    forEachWordHash(sentence, [&](uint32_t word_hash) {
      unigrams.push_back(word_hash % output_range);
    });
    return unigrams;
  }

  static std::vector<uint32_t> computeUnigrams(
      const std::string_view sentence) {
    std::vector<uint32_t> unigrams;
    forEachWordHash(sentence, [&](uint32_t word_hash) {
      unigrams.push_back(word_hash);
      return unigrams;
    });
  }

  static bolt::BoltVector computePairgramsFromUnigrams(
      const std::vector<uint32_t>& unigram_hashes, uint32_t output_range) {
    std::unordered_map<uint32_t, uint32_t> pairgram_hashes;

    // Merge all ordered pairs of unigram hashes.
    for (uint32_t token = 0; token < unigram_hashes.size(); token++) {
      for (uint32_t prev_token = 0; prev_token <= token; prev_token++) {
        uint32_t combined_hash = hashing::HashUtils::combineHashes(
            unigram_hashes[prev_token], unigram_hashes[token]);
        combined_hash = combined_hash % output_range;
        pairgram_hashes[combined_hash]++;
      }
    }

    // Construct bolt vector from unique nonzeros.
    bolt::BoltVector data_vec(pairgram_hashes.size(), false, false);
    uint32_t index = 0;
    for (auto& entry : pairgram_hashes) {
      data_vec.active_neurons[index] = entry.first;
      data_vec.activations[index] = entry.second;
      index++;
    }

    return data_vec;
  }

  static bolt::BoltVector computePairgrams(std::string_view sentence,
                                           uint32_t output_range) {
    auto unigrams = computeUnigrams(sentence);
    return computePairgramsFromUnigrams(unigrams, output_range);
  }

  /**
   * Deduplicates indices by summing values and adds features to the given
   * vector. All indices expected to correspond to the same value.
   */
  inline static void sumRepeatedIndices(std::vector<uint32_t>& indices,
                                        float value,
                                        SegmentedFeatureVector& vec) {
    // Put equivalent indices next to each other.
    std::sort(indices.begin(), indices.end());

    /**
     * If current index is the same as the next index, keep accumulating
     * val. Otherwise, add sparse feature at the current index with the
     * accumulated value and reset val.
     */
    float val = 0.0;
    uint32_t i = 0;
    for (; i < indices.size() - 1; ++i) {
      uint32_t idx = indices[i];
      uint32_t next_idx = indices[i + 1];
      val += value;

      if (idx != next_idx) {
        vec.addSparseFeatureToSegment(idx, val);
        val = 0.0;  // Reset val since next idx is different.
      }
    }

    /**
     * If we're looking at the last element, the next element is clearly
     * "different", so we add a sparse feature accordingly.
     */
    if (i == indices.size() - 1) {
      val += value;
      vec.addSparseFeatureToSegment(indices.back(), val);
    }
  }

  /**
   * Creates a copy of the original string where all characters are lowercase.
   */
  inline static std::string makeLowerCase(const std::string_view original) {
    std::string lower_case_text(original);
    for (auto& c : lower_case_text) {
      c = std::tolower(c);
    }
    return lower_case_text;
  }
};

}  // namespace thirdai::dataset