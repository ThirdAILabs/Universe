#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <utils/StringManipulation.h>
#include <functional>
#include <string_view>
#include <type_traits>

namespace thirdai::dataset::token_encoding {

std::vector<uint32_t> ngrams(const std::vector<std::string_view>& words,
                             uint32_t n) {
  uint32_t num_words = words.size();

  std::vector<uint32_t> n_gram_tokens;
  n_gram_tokens.reserve(words.size());

  // - We compute unigrams first because it makes the loop below more simple
  // and makes computing just unigrams faster
  // - We include unigrams in all n-gram computations so bolt always can
  // identify keywords
  for (const auto& word : words) {
    n_gram_tokens.push_back(seededMurmurHash(word.data(), word.size()));
  }

  // if we have to do more than unigrams and there are enough words to N-gram
  if (n > 1 && n <= num_words) {
    for (uint32_t start_token_idx = 0; start_token_idx <= num_words - n;
         start_token_idx++) {
      uint32_t n_gram_token = n_gram_tokens[start_token_idx];

      for (uint32_t i = 1; i < n; i++) {
        uint32_t next_token = n_gram_tokens[start_token_idx + i];
        n_gram_token = hashing::combineHashes(n_gram_token, next_token);
      }
      n_gram_tokens.push_back(n_gram_token);
    }
  }

  return n_gram_tokens;
}

std::vector<uint32_t> pairgrams(const std::vector<uint32_t>& unigrams) {
  std::vector<uint32_t> tokens;
  for (uint32_t token = 0; token < unigrams.size(); token++) {
    for (uint32_t prev_token = 0; prev_token <= token; prev_token++) {
      tokens.push_back(
          hashing::combineHashes(unigrams[prev_token], unigrams[token]));
    }
  }

  return tokens;
}

std::vector<uint32_t> ngrams(std::string_view sentence, uint32_t n,
                             char delimiter) {
  auto words = thirdai::utils::splitIntoWords(sentence, delimiter);

  return ngrams(words, n);
}

std::vector<uint32_t> unigrams(std::string_view sentence, char delimiter) {
  return ngrams(sentence, /* n= */ 1, delimiter);
}

std::vector<uint32_t> unigrams(const std::vector<std::string_view>& words) {
  return ngrams(words, /* n= */ 1);
}

std::vector<uint32_t> pairgrams(std::string_view sentence) {
  std::vector<uint32_t> tokens = unigrams(sentence);

  return pairgrams(tokens);
}

void mod(std::vector<uint32_t>& tokens, uint32_t dim) {
  for (uint32_t& token : tokens) {
    token %= dim;
  }
}

std::unordered_map<uint32_t, std::string> buildUnigramHashToWordMap(
    std::string_view sentence, uint32_t output_range, char delimiter) {
  auto words = thirdai::utils::splitIntoWords(sentence, delimiter);

  auto tokens = unigrams(words);

  assert(words.size() == tokens.size());
  uint32_t length = words.size();

  std::unordered_map<uint32_t, std::string> index_to_word;
  for (uint32_t i = 0; i < length; i++) {
    index_to_word[tokens[i] % output_range] = words[i];
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
  if (i == indices.size() - 1) {
    summed_val += 1.0;
    index_value_pairs.push_back(std::make_pair(indices.back(), summed_val));
  }

  return index_value_pairs;
}

}  // namespace thirdai::dataset::token_encoding