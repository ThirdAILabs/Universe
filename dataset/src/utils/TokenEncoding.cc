#include "TokenEncoding.h"
#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <functional>
#include <string_view>
#include <type_traits>

namespace thirdai::dataset::TokenEncoding {

std::vector<uint32_t> computeNGrams(const std::vector<std::string_view>& words,
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
    for (uint32_t start_word_idx = 0; start_word_idx <= num_words - n;
         start_word_idx++) {
      std::string_view word = words[start_word_idx];
      uint32_t n_gram_token = seededMurmurHash(word.data(), word.size());

      for (uint32_t i = 1; i < n; i++) {
        word = words[start_word_idx + i];
        n_gram_token = hashing::HashUtils::combineHashes(
            n_gram_token, seededMurmurHash(word.data(), word.size()));
      }
      n_gram_tokens.push_back(n_gram_token);
    }
  }

  return n_gram_tokens;
}

std::vector<uint32_t> computePairGrams(const std::vector<uint32_t>& unigrams) {
  std::vector<uint32_t> pairgrams;
  for (uint32_t token = 0; token < unigrams.size(); token++) {
    for (uint32_t prev_token = 0; prev_token <= token; prev_token++) {
      pairgrams.push_back(hashing::HashUtils::combineHashes(
          unigrams[prev_token], unigrams[token]));
    }
  }

  return pairgrams;
}

std::vector<uint32_t> computeNGrams(std::string_view sentence, uint32_t n,
                                    char delimiter) {
  auto words = splitIntoWords(sentence, delimiter);

  return computeNGrams(words, n);
}

std::vector<uint32_t> computeUnigrams(std::string_view sentence,
                                      char delimiter) {
  return computeNGrams(sentence, /* n= */ 1, delimiter);
}

std::vector<uint32_t> computeUnigrams(
    const std::vector<std::string_view>& words) {
  return computeNGrams(words, /* n= */ 1);
}

std::vector<uint32_t> computePairGrams(std::string_view sentence) {
  std::vector<uint32_t> unigrams = computeUnigrams(sentence);

  return computePairGrams(unigrams);
}

void mod(std::vector<uint32_t>& tokens, uint32_t dim) {
  for (uint32_t& token : tokens) {
    token %= dim;
  }
}

std::unordered_map<uint32_t, std::string> buildUnigramHashToWordMap(
    std::string_view sentence, uint32_t output_range, char delimiter) {
  auto words = splitIntoWords(sentence, delimiter);

  auto unigrams = computeUnigrams(words);

  assert(words.size() == unigrams.size());
  uint32_t length = words.size();

  std::unordered_map<uint32_t, std::string> index_to_word;
  for (uint32_t i = 0; i < length; i++) {
    index_to_word[unigrams[i] % output_range] = words[i];
  }

  return index_to_word;
}

template <typename PAIRGRAM_PROCESSOR_T>
void forEachPairgramFromUnigram(const std::vector<uint32_t>& unigram_hashes,
                                uint32_t output_range,
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

std::vector<std::string_view> splitIntoWords(std::string_view sentence,
                                             char delimiter) {
  std::vector<std::string_view> words;

  bool prev_is_delim = true;
  uint32_t start_of_word_offset;
  for (uint32_t i = 0; i < sentence.size(); i++) {
    if (prev_is_delim && sentence[i] != delimiter) {
      // If we go from a space to a non-space character then we are at the
      // start of a word.
      start_of_word_offset = i;
      prev_is_delim = false;
    }
    if (!prev_is_delim && sentence[i] == delimiter) {
      // If we go from a non-space character to a space then we are at the end
      // of a word.
      uint32_t len = i - start_of_word_offset;

      std::string_view word_view(sentence.data() + start_of_word_offset, len);

      words.push_back(word_view);
      prev_is_delim = true;
    }
  }
  if (!prev_is_delim) {
    // If we don't find a space at the end of the sentence, then there's a
    // last word we need to hash.
    uint32_t len = sentence.size() - start_of_word_offset;

    std::string_view word_view(sentence.data() + start_of_word_offset, len);

    words.push_back(word_view);
  }

  return words;
}

std::vector<std::pair<uint32_t, float>> sumRepeatedIndices(
    std::vector<uint32_t>& indices, float base_value) {
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
    summed_val += base_value;

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
    summed_val += base_value;
    index_value_pairs.push_back(std::make_pair(indices.back(), summed_val));
  }

  return index_value_pairs;
}

}  // namespace thirdai::dataset::TokenEncoding