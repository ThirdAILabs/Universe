#pragma once

#include <hashing/src/HashUtils.h>
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

  static uint32_t computeUnigram(const char* key, uint32_t len) {
    return hashing::MurmurHash(key, len, HASH_SEED);
  }

  static std::vector<uint32_t> computeRawUnigrams(
      const std::string_view sentence) {
    std::vector<uint32_t> unigrams;
    forEachWordHash(sentence,
                    [&](uint32_t word_hash) { unigrams.push_back(word_hash); });
    return unigrams;
  }

  static std::vector<uint32_t> computeRawUnigramsWithRange(
      const std::string_view sentence, uint32_t output_range) {
    std::vector<uint32_t> unigrams;
    forEachWordHash(sentence, [&](uint32_t word_hash) {
      unigrams.push_back(word_hash % output_range);
    });
    return unigrams;
  }

  static bolt::BoltVector computeUnigrams(const std::string_view sentence,
                                          uint32_t output_range) {
    std::vector<uint32_t> unigrams =
        computeRawUnigramsWithRange(sentence, output_range);

    std::vector<uint32_t> indices;
    std::vector<float> values;

    sumRepeatedIndices(unigrams, /* value */ 1.0,
                       [&](uint32_t unigram, float value) {
                         indices.push_back(unigram);
                         values.push_back(value);
                       });

    return bolt::BoltVector::makeSparseVector(indices, values);
  }

  static std::vector<uint32_t> computeRawPairgramsFromUnigrams(
      std::vector<uint32_t> unigram_hashes, uint32_t output_range) {
    std::vector<uint32_t> pairgram_hashes;

    // Merge all ordered pairs of unigram hashes.
    for (uint32_t token = 0; token < unigram_hashes.size(); token++) {
      for (uint32_t prev_token = 0; prev_token <= token; prev_token++) {
        uint32_t combined_hash = hashing::HashUtils::combineHashes(
            unigram_hashes[prev_token], unigram_hashes[token]);
        combined_hash = combined_hash % output_range;
        pairgram_hashes.push_back(combined_hash);
      }
    }
    return pairgram_hashes;
  }

  /**
   * Compute raw pairgrams modded to a certain range.
   */
  static std::vector<uint32_t> computeRawPairgrams(std::string_view sentence,
                                                   uint32_t output_range) {
    std::vector<uint32_t> unigram_hashes = computeRawUnigrams(sentence);

    return computeRawPairgramsFromUnigrams(unigram_hashes, output_range);
  }

  /**
   * Computes pairgrams into a BoltVector
   */
  static bolt::BoltVector computePairgrams(std::string_view sentence,
                                           uint32_t output_range) {
    std::vector<uint32_t> pairgrams =
        computeRawPairgrams(sentence, output_range);

    std::vector<uint32_t> indices;
    std::vector<float> values;

    sumRepeatedIndices(pairgrams, /* value */ 1.0,
                       [&](uint32_t pairgram, float value) {
                         indices.push_back(pairgram);
                         values.push_back(value);
                       });

    return bolt::BoltVector::makeSparseVector(indices, values);
  }

  static bolt::BoltVector computePairgramsFromUnigrams(
      std::vector<uint32_t>& unigrams, uint32_t output_range) {
    std::vector<uint32_t> pairgrams =
        computeRawPairgramsFromUnigrams(unigrams, output_range);

    uint32_t index = 0;
    bolt::BoltVector data_vec(pairgrams.size(), false, false);

    sumRepeatedIndices(pairgrams, /* value */ 1.0,
                       [&](uint32_t pairgram, float value) {
                         data_vec.active_neurons[index] = pairgram;
                         data_vec.activations[index] = value;
                         index++;
                       });
    return data_vec;
  }

  /**
   * Parses through a sentence and applies a function to the hash of each
   * word.
   */
  template <typename WORD_PROCESSOR_T>
  inline static void forEachWordHash(const std::string_view sentence,
                                     WORD_PROCESSOR_T word_processor) {
    static_assert(std::is_convertible<WORD_PROCESSOR_T,
                                      std::function<void(uint32_t)>>::value);

    bool prev_is_space = true;
    uint32_t start_of_word_offset;
    for (uint32_t i = 0; i < sentence.size(); i++) {
      if (prev_is_space && !std::isspace(sentence[i])) {
        // If we go from a space to a non-space character then we are at the
        // start of a word.
        start_of_word_offset = i;
        prev_is_space = false;
      }
      if (!prev_is_space && std::isspace(sentence[i])) {
        // If we go from a non-space character to a space then we are at the end
        // of a word.
        uint32_t len = i - start_of_word_offset;

        // Hash the word using the recorded start offset and the current index.
        uint32_t word_hash =
            computeUnigram(sentence.data() + start_of_word_offset, len);
        word_processor(word_hash);
        prev_is_space = true;
      }
    }
    if (!prev_is_space) {
      // If we don't find a space at the end of the sentence, then there's a
      // last word we need to hash.
      uint32_t len = sentence.size() - start_of_word_offset;
      uint32_t word_hash =
          computeUnigram(sentence.data() + start_of_word_offset, len);
      word_processor(word_hash);
    }
  }

  /**
   * Sorts the given indices and deduplicates them by adding value for each
   * instance of that index. Applies a lambda to the resulting idx, value pair.
   */
  template <typename INDEX_VAL_PROCESSOR>
  static void sumRepeatedIndices(std::vector<uint32_t>& indices, float value,
                                 INDEX_VAL_PROCESSOR idx_val_processor) {
    static_assert(
        std::is_convertible<INDEX_VAL_PROCESSOR,
                            std::function<void(uint32_t, float)>>::value);

    std::sort(indices.begin(), indices.end());

    std::vector<uint32_t> new_indices;

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
        idx_val_processor(idx, val);
        val = 0.0;  // Reset val since next idx is different.
      }
    }

    /**
     * If we're looking at the last element, the next element is clearly
     * "different", so we add a sparse feature accordingly.
     */
    if (i == indices.size() - 1) {
      val += value;
      idx_val_processor(indices.back(), val);
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