#pragma once

#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <functional>
#include <string_view>
#include <type_traits>

namespace thirdai::dataset {

/**
 * This class should be the source of all text encodings in the codebase to
 * ensure no mismatches in hashes or implementations.
 */
class TextEncodingUtils {
 public:
  static constexpr uint32_t HASH_SEED = 341;
  static constexpr uint32_t DEFAULT_TEXT_ENCODING_DIM = 100000;

  static uint32_t computeUnigram(const char* key, uint32_t len) {
    return hashing::MurmurHash(key, len, HASH_SEED);
  }

  /**
   * Unigrams in a vector with possible repeated indices
   */
  static std::vector<uint32_t> computeRawUnigrams(
      const std::string_view sentence) {
    std::vector<uint32_t> unigrams;
    forEachWordHash(sentence,
                    [&](uint32_t word_hash, const std::string_view& word) {
                      (void)word;
                      unigrams.push_back(word_hash);
                    });
    return unigrams;
  }

  /**
   * Unigrams in a vector with possible repeated indices (modded to a range)
   */
  static std::vector<uint32_t> computeRawUnigramsWithRange(
      const std::string_view sentence, uint32_t output_range) {
    std::vector<uint32_t> unigrams;
    forEachWordHash(sentence,
                    [&](uint32_t word_hash, const std::string_view& word) {
                      (void)word;
                      unigrams.push_back(word_hash % output_range);
                    });
    return unigrams;
  }

  /**
   * Get the word_hash to word map, which we can use it for RCA. Its better to
   * write seperate function than to overload the already existing function.
   */
  static std::unordered_map<uint32_t, std::string> buildUnigramHashToWordMap(
      const std::string_view sentence, uint32_t output_range) {
    std::unordered_map<uint32_t, std::string> index_to_word;
    forEachWordHash(sentence,
                    [&](uint32_t word_hash, const std::string_view& word) {
                      (void)word_hash;
                      index_to_word[word_hash % output_range] = word;
                    });
    return index_to_word;
  }

  /**
   * Unigrams in a BoltVector with possible repeated indices summed up
   */
  static BoltVector computeUnigrams(const std::string_view sentence,
                                    uint32_t output_range) {
    std::vector<uint32_t> unigrams =
        computeRawUnigramsWithRange(sentence, output_range);

    std::vector<uint32_t> indices;
    std::vector<float> values;

    sumRepeatedIndices(unigrams, /* base_value= */ 1.0,
                       [&](uint32_t unigram, float value) {
                         indices.push_back(unigram);
                         values.push_back(value);
                       });

    return BoltVector::makeSparseVector(indices, values);
  }

  struct PairGram {
    uint32_t pairgram;
    uint32_t first_token;
    uint32_t second_token;
  };

  template <typename PAIRGRAM_PROCESSOR_T>
  static void forEachPairgramFromUnigram(
      const std::vector<uint32_t>& unigram_hashes, uint32_t output_range,
      PAIRGRAM_PROCESSOR_T pairgram_processor) {
    static_assert(std::is_convertible<PAIRGRAM_PROCESSOR_T,
                                      std::function<void(PairGram)>>::value);

    for (uint32_t token = 0; token < unigram_hashes.size(); token++) {
      for (uint32_t prev_token = 0; prev_token <= token; prev_token++) {
        uint32_t combined_hash = hashing::HashUtils::combineHashes(
            unigram_hashes[prev_token], unigram_hashes[token]);
        combined_hash = combined_hash % output_range;
        pairgram_processor({/* pairgram= */ combined_hash,
                            /* first_token= */ unigram_hashes[prev_token],
                            /* second_token= */ unigram_hashes[token]});
      }
    }
  }

  /**
   * Pairgrams in a vector with possible repeated indices
   */
  static std::vector<uint32_t> computeRawPairgramsFromUnigrams(
      const std::vector<uint32_t>& unigram_hashes, uint32_t output_range) {
    std::vector<uint32_t> pairgram_hashes;

    forEachPairgramFromUnigram(unigram_hashes, output_range,
                               [&](PairGram pairgram) {
                                 pairgram_hashes.push_back(pairgram.pairgram);
                               });
    return pairgram_hashes;
  }

  /**
   * Pairgrams in a vector with possible repeated indices
   */
  static std::vector<uint32_t> computeRawPairgrams(std::string_view sentence,
                                                   uint32_t output_range) {
    std::vector<uint32_t> unigram_hashes = computeRawUnigrams(sentence);

    return computeRawPairgramsFromUnigrams(unigram_hashes, output_range);
  }

  /**
   * Pairgrams in a BoltVector with possible repeated indices summed up
   */
  static BoltVector computePairgrams(std::string_view sentence,
                                     uint32_t output_range) {
    std::vector<uint32_t> pairgrams =
        computeRawPairgrams(sentence, output_range);

    std::vector<uint32_t> indices;
    std::vector<float> values;

    sumRepeatedIndices(pairgrams, /* base_value= */ 1.0,
                       [&](uint32_t pairgram, float value) {
                         indices.push_back(pairgram);
                         values.push_back(value);
                       });

    return BoltVector::makeSparseVector(indices, values);
  }

  /**
   * Pairgrams in a BoltVector with possible repeated indices summed up
   */
  static BoltVector computePairgramsFromUnigrams(
      std::vector<uint32_t>& unigrams, uint32_t output_range) {
    std::vector<uint32_t> pairgrams =
        computeRawPairgramsFromUnigrams(unigrams, output_range);

    std::vector<uint32_t> indices;
    std::vector<float> values;

    sumRepeatedIndices(pairgrams, /* base_value= */ 1.0,
                       [&](uint32_t pairgram, float value) {
                         indices.push_back(pairgram);
                         values.push_back(value);
                       });

    return BoltVector::makeSparseVector(indices, values);
  }

  /**
   * Parses through a sentence and applies a function to the hash of each
   * word.
   */
  template <typename WORD_PROCESSOR_T>
  inline static void forEachWordHash(const std::string_view sentence,
                                     WORD_PROCESSOR_T word_processor) {
    static_assert(std::is_convertible<
                  WORD_PROCESSOR_T,
                  std::function<void(uint32_t, std::string_view)>>::value);

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

        std::string_view word_view(sentence.data() + start_of_word_offset, len);

        // Hash the word using the recorded start offset and the current index.
        uint32_t word_hash =
            computeUnigram(sentence.data() + start_of_word_offset, len);
        word_processor(word_hash, word_view);
        prev_is_space = true;
      }
    }
    if (!prev_is_space) {
      // If we don't find a space at the end of the sentence, then there's a
      // last word we need to hash.
      uint32_t len = sentence.size() - start_of_word_offset;

      std::string_view word_view(sentence.data() + start_of_word_offset, len);

      uint32_t word_hash =
          computeUnigram(sentence.data() + start_of_word_offset, len);
      word_processor(word_hash, word_view);
    }
  }

  /**
   * Sorts the given indices and deduplicates them by adding base_value for each
   * instance of that index. Applies a lambda to the resulting idx, summed_value
   * pair.
   */
  template <typename INDEX_VAL_PROCESSOR>
  static void sumRepeatedIndices(std::vector<uint32_t>& indices,
                                 float base_value,
                                 INDEX_VAL_PROCESSOR idx_val_processor) {
    static_assert(
        std::is_convertible<INDEX_VAL_PROCESSOR,
                            std::function<void(uint32_t, float)>>::value);

    if (indices.empty()) {
      return;
    }

    std::sort(indices.begin(), indices.end());

    /**
     * If current index is the same as the next index, keep accumulating
     * summed_val. Otherwise, add sparse feature at the current index with the
     * accumulated base_value and reset summed_val.
     */
    float summed_val = 0.0;
    uint32_t i = 0;
    for (; i < indices.size() - 1; ++i) {
      uint32_t idx = indices[i];
      uint32_t next_idx = indices[i + 1];
      summed_val += base_value;

      if (idx != next_idx) {
        idx_val_processor(idx, summed_val);
        summed_val = 0.0;  // Reset summed_val since next idx is different.
      }
    }

    /**
     * If we're looking at the last element, the next element is clearly
     * "different", so we add a sparse feature accordingly.
     */
    if (i == indices.size() - 1) {
      summed_val += base_value;
      idx_val_processor(indices.back(), summed_val);
    }
  }
};

}  // namespace thirdai::dataset