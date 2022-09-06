#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <gtest/gtest.h>
#include <dataset/src/StreamingDataset.h>
#include <dataset/src/batch_processors/MaskedSentenceBatchProcessor.h>
#include <dataset/src/batch_processors/TextClassificationProcessor.h>
#include <dataset/src/encodings/text/TextEncodingUtils.h>
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

void testCreateBatchArbitraryLabels(
    const std::vector<std::string>& rows, const std::string& header,
    std::vector<std::vector<std::string>> words) {
  TextClassificationProcessor processor(RANGE);

  processor.processHeader(header);

  auto [data, labels] = processor.createBatch(rows);

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

  std::vector<std::vector<std::string>> words = {
      {"tasty", "red", "fruit"},
      {"green", "fruit", "in", "the", "fall"},
      {"make", "delicious", "pies"},
      {"grow", "on", "trees"}};

  testCreateBatchArbitraryLabels(rows, header, words);
}

TEST(TextClassifierBatchProcessor, TestCreateBatchLabelsRight) {
  std::vector<std::string> rows = {
      R"("tasty red fruit","apple")", R"( green fruit in the fall", 'pear')",
      R"(' make delicious pies, apple)", R"(grow on trees ' ,"mango' )"};

  std::string header = R"("text","category")";

  std::vector<std::vector<std::string>> words = {
      {"tasty", "red", "fruit"},
      {"green", "fruit", "in", "the", "fall"},
      {"make", "delicious", "pies"},
      {"grow", "on", "trees"}};

  testCreateBatchArbitraryLabels(rows, header, words);
}

TEST(MaskedSentenceBatchProcessor, TestCreateBatch) {
  std::vector<std::string> rows = {
      "the dog ran up the hill", "the cat slept on the window",
      "the rhino has a horn", "the monkey climbed the tree"};

  std::vector<std::vector<std::string>> words{
      {"the", "dog", "ran", "up", "the", "hill"},
      {"the", "cat", "slept", "on", "the", "window"},
      {"the", "rhino", "has", "a", "horn"},
      {"the", "monkey", "climbed", "the", "tree"}};

  dataset::MaskedSentenceBatchProcessor processor(RANGE);

  auto [data, masked_indices, labels] = processor.createBatch(rows);

  uint32_t unknown_hash =
      TextEncodingUtils::computeUnigram(/* key= */ "[UNK]", /* len= */ 5);

  const std::unordered_map<uint32_t, uint32_t>& words_to_ids =
      processor.getWordToIDMap();

  std::unordered_set<uint32_t> masked_word_hashes;

  EXPECT_EQ(data.getBatchSize(), 4);
  EXPECT_EQ(masked_indices.getBatchSize(), 4);
  EXPECT_EQ(labels.getBatchSize(), 4);

  for (uint32_t i = 0; i < 4; i++) {
    auto unigrams = unigram_hashes_from_words(words[i]);
    uint32_t masked_index = masked_indices[i].active_neurons[0];
    uint32_t masked_word_hash = unigrams[masked_index];
    unigrams[masked_index] = unknown_hash;

    auto pairgrams = pairgram_hashes_as_map(unigrams, RANGE);

    checkPairgramVector(data[i], pairgrams);

    uint32_t label = labels[i].active_neurons[0];
    ASSERT_EQ(label, words_to_ids.at(masked_word_hash));

    masked_word_hashes.insert(masked_word_hash);
  }

  // Verify that we have the correct number of tokens of masked words.
  ASSERT_EQ(words_to_ids.size(), masked_word_hashes.size());

  // Check that word ids are distinct.
  std::unordered_set<uint32_t> masked_word_ids;
  for (const auto& [k, v] : words_to_ids) {
    masked_word_ids.insert(v);
  }
  ASSERT_EQ(words_to_ids.size(), masked_word_ids.size());
}

TEST(MaskedSentenceBatchProcessor, TestCreateBatchMultipleMaskedTokens) {
  std::vector<std::string> rows{
      "the dog ran up the hill and came back down right away",
      "the cat slept on the window for a very long time",
      "the monkey and the rhino were playing outside when the monkey got "
      "injured",
      "we all love natural language processing and computer vision"};

  std::vector<std::vector<std::string>> split_sentences{
      {"the", "dog", "ran", "up", "the", "hill", "and", "came", "back", "down",
       "right", "away"},
      {"the", "cat", "slept", "on", "the", "window", "for", "a", "very", "long",
       "time"},
      {"the", "monkey", "and", "the", "rhino", "were", "playing", "outside",
       "when", "the", "monkey", "got", "injured"},
      {"we", "all", "love", "natural", "language", "processing", "and",
       "computer", "vision"}};

  dataset::MaskedSentenceBatchProcessor processor(
      RANGE, /* masked_tokens_percentage= */ 0.3);

  auto [data, masked_indices, labels] = processor.createBatch(rows);

  uint32_t unknown_hash =
      TextEncodingUtils::computeUnigram(/* key= */ "[UNK]", /* len= */ 5);

  const std::unordered_map<uint32_t, uint32_t>& words_to_ids =
      processor.getWordToIDMap();

  EXPECT_EQ(data.getBatchSize(), 4);
  EXPECT_EQ(masked_indices.getBatchSize(), 4);
  EXPECT_EQ(labels.getBatchSize(), 4);

  for (uint32_t index = 0; index < 4; index++) {
    BoltVector label_vector = labels[index];
    auto unigrams = unigram_hashes_from_words(split_sentences[index]);

    ASSERT_EQ(masked_indices[index].len,
              static_cast<uint32_t>(unigrams.size() * 0.3));

    for (uint32_t i = 0; i < label_vector.len; i++) {
      uint32_t masked_index = masked_indices[index].active_neurons[i];
      uint32_t masked_word_hash = unigrams[masked_index];

      unigrams[masked_index] = unknown_hash;

      ASSERT_EQ(label_vector.active_neurons[i],
                words_to_ids.at(masked_word_hash));
    }
    auto pairgrams = pairgram_hashes_as_map(unigrams, RANGE);

    checkPairgramVector(data[index], pairgrams);
  }
}

}  // namespace thirdai::dataset::tests