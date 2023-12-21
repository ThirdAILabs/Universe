#pragma once

#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <utils/text/StringManipulation.h>
#include <functional>
#include <type_traits>
#include <unordered_map>

/**
 * This namespace should be the source of token encodings across the
 * codebase to ensure consistency in code, implementation, hash seeds, etc.
 */
namespace thirdai::dataset::token_encoding {

static constexpr uint32_t HASH_SEED = 341;
static constexpr uint32_t DEFAULT_TEXT_ENCODING_DIM = 100000;

inline uint32_t seededMurmurHash(const char* key, uint32_t len) {
  return hashing::MurmurHash(key, len, HASH_SEED);
}

std::vector<uint32_t> tokenIds(const std::string& line);

/**
 * Hash each input word and return a list of tokens. Commonly called unigrams.
 */
std::vector<uint32_t> hashTokens(const std::vector<std::string>& strings);

/**
 * Takes in a list of hashed tokens and uses our combineHashes function to add
 * in additional N-gram hashed tokens. If you have a vector of words you'd like
 * to convert into tokens, please use the tokenize method.
 */
std::vector<uint32_t> ngrams(std::vector<uint32_t> tokens, uint32_t n);

inline std::vector<uint32_t> ngrams(const std::string_view& sentence,
                                    uint32_t n, char delimiter = ' ') {
  return ngrams(hashTokens(text::split(sentence, delimiter)), /* n= */ n);
}

/**
 * Given a vector of unigram tokens, compute all ordered pairs of tokens and
 * combine their hashes into new tokens.
 */
std::vector<uint32_t> pairgrams(const uint32_t* tokens, uint32_t len);

inline std::vector<uint32_t> pairgrams(const std::vector<uint32_t>& unigrams) {
  return pairgrams(unigrams.data(), unigrams.size());
}

/**
 * This function differs from our regular pairgram utility because of how it
 * handles unigrams. Our regular pairgram utility will include hash(t_i, t_i)
 * for i = [0...seq_len). However there are places where we want the unigram
 * representation of tokens within the pairgrams to be consistent with the raw
 * unigrams themselves.
 */
std::vector<uint32_t> unigramPreservingPairgrams(const uint32_t* tokens,
                                                 uint32_t len,
                                                 uint32_t unigram_range);

/**
 * Mods each of the given tokens by dim.
 */
void mod(std::vector<uint32_t>& tokens, uint32_t dim);

/**
 * Compute unigram tokens of the given words, mod them by output_range, and
 * return a map from the unigram token value to the source word. Used in
 * explainability.
 */
std::unordered_map<uint32_t, std::string> buildUnigramHashToWordMap(
    const std::vector<std::string>& words);

/**
 * Given a vector of indices, sums repeated indices by multiplying the number of
 * occurrences by base_value. Returns a vector of index, value pairs where each
 * index is unique. This typically has a small overhead but should significantly
 * speed up any subsequent model training job since the number of non-zeros in
 * the input decreases.
 */
std::vector<std::pair<uint32_t, float>> sumRepeatedIndices(
    std::vector<uint32_t>& indices);

}  // namespace thirdai::dataset::token_encoding