#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <gtest/gtest.h>
#include <dataset/src/bolt_datasets/StreamingDataset.h>
#include <dataset/src/bolt_datasets/batch_processors/TextClassificationProcessor.h>
#include <unordered_map>

namespace thirdai::dataset::tests {

std::unordered_map<uint32_t, uint32_t> pairgram_hashes(
    const std::vector<std::string>& words, uint32_t range) {
  std::unordered_map<uint32_t, uint32_t> pairgrams;
  std::vector<uint32_t> word_hashes;

  for (const auto& word : words) {
    uint32_t hash = hashing::MurmurHash(word.data(), word.size(), 3829);
    for (uint32_t h : word_hashes) {
      uint32_t ch = hashing::HashUtils::combineHashes(h, hash) % range;
      pairgrams[ch]++;
    }
    word_hashes.push_back(hash);
  }

  return pairgrams;
}

TEST(TextClassifierBatchProcessor, TestComputePairgrams) {
  std::string sentence = "the red dog ran up the hill";
  std::vector<std::string> words = {"the", "red", "dog", "ran",
                                    "up",  "the", "hill"};

  TextClassificationProcessor _processor(10000);

  bolt::BoltVector vector = _processor.computePairGramHashes(sentence);

  auto pairgrams = pairgram_hashes(words, 10000);

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
