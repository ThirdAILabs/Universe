#pragma once

#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <functional>
#include <string_view>
#include <type_traits>

// todo unit tests
// todo integrate into UDT?

/**
 * @brief This namespace should be the source of token encodings across the
 * codebase to ensure consistency in code, implementation, hash seeds, etc.
 */
namespace thirdai::dataset::TokenEncoding {

static constexpr uint32_t HASH_SEED = 341;
static constexpr uint32_t DEFAULT_TEXT_ENCODING_DIM = 100000;

inline uint32_t seededMurmurHash(const char* key, uint32_t len) {
  return hashing::MurmurHash(key, len, HASH_SEED);
}

/**
 * Given a sequence of words compute all N-gram tokens (depending on the
 * supplied n value) and hash them. These are commonly referred to as
 * "unigrams", "bigrams", "trigrams", etc (not to be confused with "pairgrams"
 * which are slightly different and explained elsewhere).
 * "Unigrams" are a specific instance of N-grams where N=1, "bigrams" where N=2,
 * and so on.
 */
std::vector<uint32_t> computeNGrams(const std::vector<std::string_view>& words,
                                    uint32_t n);

/**
 * Given a vector of unigram tokens, compute all ordered pairs of tokens and
 * combine their hashes into new tokens.
 */
std::vector<uint32_t> computePairGrams(const std::vector<uint32_t>& unigrams);

std::vector<uint32_t> computeNGrams(std::string_view sentence, uint32_t n,
                                    char delimiter = ' ');

std::vector<uint32_t> computeUnigrams(std::string_view sentence,
                                      char delimiter = ' ');

std::vector<uint32_t> computeUnigrams(
    const std::vector<std::string_view>& words);

std::vector<uint32_t> computePairGrams(std::string_view sentence);

/**
 * Mods each of the given tokens by dim.
 */
void mod(std::vector<uint32_t>& tokens, uint32_t dim);

/**
 * Compute unigram tokens of a given sentence, mod them by output_range, and
 * return a map from the unigram token value to the source word. Used in
 * explainability.
 */
std::unordered_map<uint32_t, std::string> buildUnigramHashToWordMap(
    std::string_view sentence, uint32_t output_range, char delimiter = ' ');

struct PairGram {
  uint32_t pairgram;
  uint32_t first_token;
  uint32_t second_token;
};

/**
 * Given unigram hash tokens, applies a given lambda to all pairgrams
 */
template <typename PAIRGRAM_PROCESSOR_T>
void forEachPairgramFromUnigram(const std::vector<uint32_t>& unigram_hashes,
                                uint32_t output_range,
                                PAIRGRAM_PROCESSOR_T pairgram_processor);

/**
 * Splits a sentence into words by delimiter.
 */
std::vector<std::string_view> splitIntoWords(std::string_view sentence,
                                             char delimiter = ' ');

/**
 * Given a vector of indices, sums repeated indices by multiplying the number of
 * occurrences by base_value. Returns a vector of index, value pairs where each
 * index is unique. This typically has a small overhead but should significantly
 * speed up any subsequent model training job since the number of non-zeros in
 * the input decreases.
 */
std::vector<std::pair<uint32_t, float>> sumRepeatedIndices(
    std::vector<uint32_t>& indices, float base_value = 1.0);

}  // namespace thirdai::dataset::TokenEncoding