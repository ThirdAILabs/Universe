#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <gtest/gtest.h>
#include <dataset/src/bolt_datasets/StreamingDataset.h>
#include <dataset/src/bolt_datasets/batch_processors/TextClassificationProcessor.h>
#include <unordered_map>

namespace thirdai::dataset::tests {

constexpr uint32_t RANGE = 10000;

std::unordered_map<uint32_t, uint32_t> pairgram_hashes(
    const std::vector<std::string>& words, uint32_t range) {
  std::unordered_map<uint32_t, uint32_t> pairgrams;
  std::vector<uint32_t> word_hashes;

  for (const auto& word : words) {
    uint32_t hash = hashing::MurmurHash(word.data(), word.size(), 3829);
    word_hashes.push_back(hash);

    for (uint32_t h : word_hashes) {
      uint32_t ch = hashing::HashUtils::combineHashes(h, hash) % range;
      pairgrams[ch]++;
    }
  }

  return pairgrams;
}

void checkPairgramVector(const bolt::BoltVector& vector,
                         const std::vector<std::string>& words) {
  auto pairgrams = pairgram_hashes(words, RANGE);

  ASSERT_EQ(vector.len, pairgrams.size());

  for (uint32_t i = 0; i < vector.len; i++) {
    ASSERT_TRUE(pairgrams.count(vector.active_neurons[i]));
    float cnt = pairgrams.at(vector.active_neurons[i]);
    ASSERT_EQ(vector.activations[i], cnt);

    pairgrams.erase(vector.active_neurons[i]);
  }
  ASSERT_EQ(pairgrams.size(), 0);
}

TEST(TextClassifierBatchProcessor, TestComputePairgrams) {
  std::string sentence = "the red dog ran up the hill";
  std::vector<std::string> words = {"the", "red", "dog", "ran",
                                    "up",  "the", "hill"};

  TextClassificationProcessor processor(RANGE);

  bolt::BoltVector vector = processor.computePairGramHashes(sentence);

  checkPairgramVector(vector, words);
}

void testCreateBatchArbitraryLabels(const std::vector<std::string>& rows,
                                    const std::string& header) {
  std::vector<std::vector<std::string>> words = {
      {"tasty", "red", "fruit"},
      {"green", "fruit", "in", "the", "fall"},
      {"make", "delicious", "pies"},
      {"grow", "on", "trees"}};

  TextClassificationProcessor processor(RANGE);

  processor.processHeader(header);

  auto batch = processor.createBatch(rows);

  ASSERT_TRUE(batch.has_value());

  const bolt::BoltBatch& data = batch->first;
  const bolt::BoltBatch& labels = batch->second;

  ASSERT_EQ(data.getBatchSize(), 4);
  ASSERT_EQ(labels.getBatchSize(), 4);

  std::vector<uint32_t> expected_labels = {0, 1, 0, 2};

  for (uint32_t vec = 0; vec < data.getBatchSize(); vec++) {
    checkPairgramVector(data[vec], words[vec]);

    ASSERT_EQ(labels[vec].len, 1);
    ASSERT_EQ(labels[vec].active_neurons[0], expected_labels[vec]);
    ASSERT_EQ(labels[vec].activations[0], 1.0);
  }
}

TEST(TextClassifierBatchProcessor, TestCreateBatchLabelsLeft) {
  std::vector<std::string> rows = {
      R"("apple","tasty red fruit")", R"('pear', green fruit in the fall")",
      R"(apple,' make delicious pies)", R"("mango',grow on trees ' )"};

  std::string header = R"("category","text")";

  testCreateBatchArbitraryLabels(rows, header);
}

TEST(TextClassifierBatchProcessor, TestCreateBatchLabelsRight) {
  std::vector<std::string> rows = {
      R"("tasty red fruit","apple")", R"( green fruit in the fall", 'pear')",
      R"(' make delicious pies, apple)", R"(grow on trees ' ,"mango' )"};

  std::string header = R"("text","category")";

  testCreateBatchArbitraryLabels(rows, header);
}

}  // namespace thirdai::dataset::tests
