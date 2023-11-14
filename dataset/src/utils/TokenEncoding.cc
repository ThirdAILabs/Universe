#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <utils/StringManipulation.h>
#include <functional>
#include <type_traits>

namespace thirdai::dataset::token_encoding {

std::vector<uint32_t> ngrams(std::vector<uint32_t> tokens, uint32_t n) {
  uint32_t num_tokens = tokens.size();

  // if we have to do more than unigrams and there are enough words to N-gram
  if (n > 1 && n <= num_tokens) {
    for (uint32_t start_token_idx = 0; start_token_idx <= num_tokens - n;
         start_token_idx++) {
      uint32_t n_gram_token = tokens[start_token_idx];

      for (uint32_t i = 1; i < n; i++) {
        uint32_t next_token = tokens[start_token_idx + i];
        n_gram_token = hashing::combineHashes(n_gram_token, next_token);
      }
      tokens.push_back(n_gram_token);
    }
  }

  return tokens;
}

std::vector<uint32_t> tokenIds(const std::string& line) {
  std::vector<uint32_t> tokens;
  const char* start = line.data();
  const char* line_end = line.data() + line.size();
  while (start != line_end) {
    char* end;
    tokens.push_back(std::strtoul(start, &end, /* base= */ 10));
    start = end;
  }

  return tokens;
}

std::vector<uint32_t> hashTokens(const std::vector<std::string>& strings) {
  std::vector<uint32_t> hashes;
  hashes.reserve(strings.size());

  for (const auto& string : strings) {
    hashes.push_back(seededMurmurHash(string.data(), string.size()));
  }

  return hashes;
}

std::vector<uint32_t> pairgrams(const uint32_t* tokens, uint32_t len) {
  std::vector<uint32_t> pairgram_tokens;
  for (uint32_t token = 0; token < len; token++) {
    for (uint32_t prev_token = 0; prev_token <= token; prev_token++) {
      pairgram_tokens.push_back(
          hashing::combineHashes(tokens[prev_token], tokens[token]));
    }
  }

  return pairgram_tokens;
}

std::vector<uint32_t> unigramPreservingPairgrams(const uint32_t* tokens,
                                                 uint32_t len,
                                                 uint32_t unigram_range) {
  std::vector<uint32_t> pairgrams(tokens, tokens + len);
  for (uint32_t i = 0; i < len; i++) {
    for (uint32_t j = 0; j < i; j++) {
      uint32_t pairgram = hashing::combineHashes(tokens[j], tokens[i]);
      // Shift the pairgrams so that the unigrams and pairgrams are in a
      // disjoint ranges.
      // In the output unigrams are in the range [0, unigram_range)
      // and pairgrams are in the range [unigram range, UINT_MAX)
      pairgram =
          pairgram % (std::numeric_limits<uint32_t>::max() - unigram_range);
      pairgrams.push_back(pairgram + unigram_range);
    }
  }

  return pairgrams;
}

void mod(std::vector<uint32_t>& tokens, uint32_t dim) {
  for (uint32_t& token : tokens) {
    token %= dim;
  }
}

std::unordered_map<uint32_t, std::string> buildUnigramHashToWordMap(
    const std::vector<std::string>& words) {
  auto tokens = hashTokens(words);

  assert(words.size() == tokens.size());
  uint32_t length = words.size();

  std::unordered_map<uint32_t, std::string> index_to_word;
  for (uint32_t i = 0; i < length; i++) {
    index_to_word[tokens[i]] = words[i];
  }

  return index_to_word;
}

std::vector<std::pair<uint32_t, float>> sumRepeatedIndices(
    std::vector<uint32_t>& indices) {
  if (indices.empty()) {
    return {};
  }

  std::sort(indices.begin(), indices.end());

  std::vector<std::pair<uint32_t, float>> index_value_pairs;

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
    summed_val += 1.0;

    if (idx != next_idx) {
      index_value_pairs.push_back(std::make_pair(idx, summed_val));
      summed_val = 0.0;  // Reset summed_val since next idx is different.
    }
  }

  /**
   * If we're looking at the last element, the next element is clearly
   * "different", so we add a sparse feature accordingly.
   */
  summed_val += 1.0;
  index_value_pairs.push_back(std::make_pair(indices.back(), summed_val));

  return index_value_pairs;
}

}  // namespace thirdai::dataset::token_encoding