#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <gtest/gtest.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/MaskedSentenceFeaturizer.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <unordered_map>

namespace thirdai::dataset::tests {

constexpr uint32_t RANGE = 10000;

std::shared_ptr<Vocabulary> vocab_from_tokens(
    const std::vector<std::vector<std::string>>& tokenized_sentences) {
  std::stringstream vocab_stream;

  for (const auto& tokens : tokenized_sentences) {
    for (const auto& token : tokens) {
      vocab_stream << token << "\n";
    }
  }

  std::istringstream vocab_istream(vocab_stream.str(), std::ios_base::in);
  return std::make_shared<FixedVocabulary>(vocab_istream);
}

std::vector<uint32_t> ids_from_words(const std::shared_ptr<Vocabulary>& vocab,
                                     const std::vector<std::string>& words) {
  std::vector<uint32_t> ids;
  ids.reserve(words.size());
  for (const auto& word : words) {
    ids.push_back(vocab->id(word));
  }
  return ids;
}

TEST(MaskedSentenceFeaturizer, TestCreateBatch) {
  std::vector<std::string> rows = {
      "the dog ran up the hill", "the cat slept on the window",
      "the rhino has a horn", "the monkey climbed the tree"};

  std::vector<std::vector<std::string>> tokenized_sentences{
      {"the", "dog", "ran", "up", "the", "hill"},
      {"the", "cat", "slept", "on", "the", "window"},
      {"the", "rhino", "has", "a", "horn"},
      {"the", "monkey", "climbed", "the", "tree"}};

  std::shared_ptr<Vocabulary> vocab = vocab_from_tokens(tokenized_sentences);

  dataset::MaskedSentenceFeaturizer processor(vocab, RANGE);

  auto datasets = processor.createBatch(rows);
  auto data = datasets.at(0);
  auto masked_indices = datasets.at(1);
  auto labels = datasets.at(2);

  std::unordered_set<uint32_t> masked_word_hashes;

  EXPECT_EQ(data.getBatchSize(), 4);
  EXPECT_EQ(masked_indices.getBatchSize(), 4);
  EXPECT_EQ(labels.getBatchSize(), 4);

  for (uint32_t i = 0; i < 4; i++) {
    auto unigrams = ids_from_words(vocab, tokenized_sentences[i]);
    uint32_t masked_index = masked_indices[i].active_neurons[0];
    uint32_t label = labels[i].active_neurons[0];
    ASSERT_EQ(label, unigrams.at(masked_index));
  }
}

TEST(MaskedSentenceFeaturizer, TestCreateBatchMultipleMaskedTokens) {
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

  std::shared_ptr<Vocabulary> vocab = vocab_from_tokens(split_sentences);

  dataset::MaskedSentenceFeaturizer processor(
      vocab, RANGE, /* masked_tokens_percentage= */ 0.3);

  auto datasets = processor.createBatch(rows);
  auto data = datasets.at(0);
  auto masked_indices = datasets.at(1);
  auto labels = datasets.at(2);

  EXPECT_EQ(data.getBatchSize(), 4);
  EXPECT_EQ(masked_indices.getBatchSize(), 4);
  EXPECT_EQ(labels.getBatchSize(), 4);

  for (uint32_t index = 0; index < 4; index++) {
    // Only unit-test here is that percentage works.
    auto unigrams = ids_from_words(vocab, split_sentences[index]);
    ASSERT_EQ(masked_indices[index].len,
              static_cast<uint32_t>(unigrams.size() * 0.3));
    for (uint32_t j = 0; j < masked_indices[index].len; j++) {
      uint32_t masked_index = masked_indices[index].active_neurons[j];
      uint32_t label = labels[index].active_neurons[j];
      ASSERT_EQ(label, unigrams.at(masked_index));
    }
  }
}

}  // namespace thirdai::dataset::tests
