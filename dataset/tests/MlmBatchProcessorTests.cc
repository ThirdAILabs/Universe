#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <gtest/gtest.h>
#include <dataset/src/StreamingDataset.h>
#include <dataset/src/batch_processors/MaskedSentenceBatchProcessor.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <unordered_map>

namespace thirdai::dataset::tests {

constexpr uint32_t RANGE = 10000;

std::vector<uint32_t> unigram_hashes_from_words(
    const std::vector<std::string>& words) {
  std::vector<uint32_t> hashes;
  hashes.reserve(words.size());
  for (const auto& word : words) {
    hashes.push_back(
        TextEncodingUtils::computeUnigram(word.data(), word.size()));
  }
  return hashes;
}

std::unordered_map<uint32_t, uint32_t> pairgram_hashes_as_map(
    const std::vector<uint32_t>& unigram_hashes, uint32_t range) {
  std::unordered_map<uint32_t, uint32_t> pairgrams;

  for (uint32_t token = 0; token < unigram_hashes.size(); token++) {
    for (uint32_t prev_token = 0; prev_token <= token; prev_token++) {
      uint32_t token_hash = unigram_hashes[token];
      uint32_t prev_token_hash = unigram_hashes[prev_token];

      uint32_t ch =
          hashing::HashUtils::combineHashes(prev_token_hash, token_hash) %
          range;
      pairgrams[ch]++;
    }
  }

  return pairgrams;
}

void checkPairgramVector(const BoltVector& vector,
                         const std::vector<std::string>& words) {
  auto unigrams = unigram_hashes_from_words(words);
  auto pairgrams = pairgram_hashes_as_map(unigrams, RANGE);

  ASSERT_EQ(vector.len, pairgrams.size());

  for (uint32_t i = 0; i < vector.len; i++) {
    ASSERT_TRUE(pairgrams.count(vector.active_neurons[i]));
    float cnt = pairgrams.at(vector.active_neurons[i]);
    ASSERT_EQ(vector.activations[i], cnt);

    pairgrams.erase(vector.active_neurons[i]);
  }
  ASSERT_EQ(pairgrams.size(), 0);
}

void checkPairgramVector(const BoltVector& vector,
                         std::unordered_map<uint32_t, uint32_t> pairgrams) {
  ASSERT_EQ(vector.len, pairgrams.size());

  for (uint32_t i = 0; i < vector.len; i++) {
    ASSERT_TRUE(pairgrams.count(vector.active_neurons[i]));
    float cnt = pairgrams.at(vector.active_neurons[i]);
    ASSERT_EQ(vector.activations[i], cnt);

    pairgrams.erase(vector.active_neurons[i]);
  }
  ASSERT_EQ(pairgrams.size(), 0);
}

}  // namespace thirdai::dataset::tests
