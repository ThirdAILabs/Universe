#include "gtest/gtest.h"
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/tests/BoltVectorTestUtils.h>
#include <hashing/src/HashUtils.h>
#include <dataset/src/featurizers/llm/TextClassificationFeaturizer.h>
#include <dataset/src/featurizers/llm/TextGenerationFeaturizer.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>

namespace thirdai::dataset::tests {

constexpr uint32_t LRC_LEN = 4, IRC_LEN = 3, SRC_LEN = 2, VOCAB_SIZE = 8;

void verifyGeneratedSamples(
    const std::vector<std::vector<thirdai::BoltVector>>& data,
    const std::vector<std::vector<std::vector<uint32_t>>>& expected_indices) {
  ASSERT_EQ(expected_indices.size(), data.size());

  uint32_t num_inputs = expected_indices.size();
  uint32_t num_samples = expected_indices.at(0).size();

  for (uint32_t input_id = 0; input_id < num_inputs; input_id++) {
    ASSERT_EQ(expected_indices.at(input_id).size(), data.at(input_id).size());

    for (uint32_t sample_id = 0; sample_id < num_samples; sample_id++) {
      const auto& vec = data.at(input_id).at(sample_id);
      const auto& expected_vec_indices =
          expected_indices.at(input_id).at(sample_id);

      ASSERT_EQ(expected_vec_indices.size(), vec.len);

      for (uint32_t i = 0; i < expected_vec_indices.size(); i++) {
        ASSERT_EQ(expected_vec_indices.at(i), vec.active_neurons[i]);
        ASSERT_EQ(vec.activations[i], 1.0);
      }
    }
  }
}

void checkDataFeaturization(
    const std::vector<std::string>& phrases,
    const std::vector<std::vector<std::vector<uint32_t>>>& expected_indices,
    bool include_position = false, bool featurize_in_chunks = true) {
  TextGenerationFeaturizer processor(
      /* lrc_len= */ LRC_LEN,
      /* irc_len= */ IRC_LEN,
      /* src_len= */ SRC_LEN,
      /* vocab_size= */ VOCAB_SIZE,
      /* include_position= */ include_position,
      /* featurize_in_chunks= */ featurize_in_chunks);

  auto data = processor.featurize(phrases);

  verifyGeneratedSamples(data, expected_indices);
}

void checkInferenceFeaturization(
    const std::vector<uint32_t>& prompt, const std::vector<uint32_t>& tokens,
    const std::vector<std::vector<std::vector<uint32_t>>>& expected_indices) {
  TextGenerationFeaturizer processor(/* lrc_len= */ LRC_LEN,
                                     /* irc_len= */ IRC_LEN,
                                     /* src_len= */ SRC_LEN,
                                     /* vocab_size= */ VOCAB_SIZE);

  auto vectors = processor.featurizeInferenceSample(prompt, tokens);

  std::vector<std::vector<BoltVector>> data;
  data.reserve(vectors.size());
  for (const auto& vector : vectors) {
    data.push_back({vector});
  }

  verifyGeneratedSamples(data, expected_indices);
}

std::vector<uint32_t> expectedPairgrams(std::vector<uint32_t> tokens) {
  return token_encoding::unigramPreservingPairgrams(tokens.data(),
                                                    tokens.size(), VOCAB_SIZE);
}

TEST(TextGenerationFeaturizerTest, FeaturizationWithChunks) {
  std::vector<std::string> phrases = {R"({"target": "1 2 3 4 5 6 7 8"})"};

  std::vector<std::vector<std::vector<uint32_t>>> expected_indices = {
      // Prompt input
      {{0}, {0}, {0}, {0}, {0}, {0}},
      //  LRC context input
      {{1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, {6}, {6, 7}},
      // IRC context input
      {
          {1},
          expectedPairgrams({1, 2}),
          expectedPairgrams({1, 2, 3}),
          expectedPairgrams({2, 3, 4}),
          {6},
          expectedPairgrams({6, 7}),
      },
      // SRC context input
      {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 6}, {6, 7}},
      // Labels
      {{2}, {3}, {4}, {5}, {7}, {8}}};

  checkDataFeaturization(phrases, expected_indices);
}

TEST(TextGenerationFeaturizerTest, FeaturizationWithSlidingWindow) {
  std::vector<std::string> phrases = {R"({"target": "1 2 3 4 5 6 7 8"})"};

  std::vector<std::vector<std::vector<uint32_t>>> expected_indices = {
      // Prompt input
      {{0}, {0}, {0}, {0}, {0}, {0}, {0}},
      //  LRC context input
      {{1},
       {1, 2},
       {1, 2, 3},
       {1, 2, 3, 4},
       {2, 3, 4, 5},
       {3, 4, 5, 6},
       {4, 5, 6, 7}},
      // IRC context input
      {
          {1},
          expectedPairgrams({1, 2}),
          expectedPairgrams({1, 2, 3}),
          expectedPairgrams({2, 3, 4}),
          expectedPairgrams({3, 4, 5}),
          expectedPairgrams({4, 5, 6}),
          expectedPairgrams({5, 6, 7}),
      },
      // SRC context input
      {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}},
      // Labels
      {{2}, {3}, {4}, {5}, {6}, {7}, {8}}};

  checkDataFeaturization(phrases, expected_indices,
                         /* include_position= */ false,
                         /* featurize_in_chunks= */ false);
}

TEST(TextGenerationFeaturizerTest, FeaturizationWithSlidingWindowContext) {
  std::vector<std::string> phrases = {
      R"({"target": "4 5 6 7 8", "context": "1 2 3"})"};

  std::vector<std::vector<std::vector<uint32_t>>> expected_indices = {
      // Prompt input
      {{0}, {0}, {0}, {0}, {0}},
      //  LRC context input
      {{1, 2, 3}, {1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}},
      // IRC context input
      {
          expectedPairgrams({1, 2, 3}),
          expectedPairgrams({2, 3, 4}),
          expectedPairgrams({3, 4, 5}),
          expectedPairgrams({4, 5, 6}),
          expectedPairgrams({5, 6, 7}),
      },
      // SRC context input
      {{2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}},
      // Labels
      {{4}, {5}, {6}, {7}, {8}}};

  checkDataFeaturization(phrases, expected_indices,
                         /* include_position= */ false,
                         /* featurize_in_chunks= */ false);
}

TEST(TextGenerationFeaturizerTest,
     FeaturizationWithSlidingWindowContextPosition) {
  std::vector<std::string> phrases = {
      R"({"target": "4 5 6 7 8", "context": "1 2 3"})"};

  std::vector<std::vector<std::vector<uint32_t>>> expected_indices = {
      // Prompt input
      {{0}, {0}, {0}, {0}, {0}},
      //  LRC context input
      {{1, 2, 3}, {1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}},
      // IRC context input
      {
          expectedPairgrams({1, 2, 3}),
          expectedPairgrams({2, 3, 4}),
          expectedPairgrams({3, 4, 5}),
          expectedPairgrams({4, 5, 6}),
          expectedPairgrams({5, 6, 7}),
      },
      // SRC context input
      {{2, 3, 11}, {3, 4, 12}, {4, 5, 13}, {5, 6, 14}, {6, 7, 15}},
      // Labels
      {{4}, {5}, {6}, {7}, {8}}};

  checkDataFeaturization(phrases, expected_indices,
                         /* include_position= */ true,
                         /* featurize_in_chunks= */ false);
}

TEST(TextGenerationFeaturizerTest,
     FeaturizationWithSlidingWindowContextPrompt) {
  std::vector<std::string> phrases = {
      R"({"target": "4 5 6 7 8", "context": "1 2 3", "prompt": "2 4 6"})"};

  std::vector<std::vector<std::vector<uint32_t>>> expected_indices = {
      // Prompt input
      {{2, 4, 6}, {2, 4, 6}, {2, 4, 6}, {2, 4, 6}, {2, 4, 6}},
      //  LRC context input
      {{1, 2, 3}, {1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}},
      // IRC context input
      {
          expectedPairgrams({1, 2, 3}),
          expectedPairgrams({2, 3, 4}),
          expectedPairgrams({3, 4, 5}),
          expectedPairgrams({4, 5, 6}),
          expectedPairgrams({5, 6, 7}),
      },
      // SRC context input
      {{2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}},
      // Labels
      {{4}, {5}, {6}, {7}, {8}}};

  checkDataFeaturization(phrases, expected_indices,
                         /* include_position= */ false,
                         /* featurize_in_chunks= */ false);
}

TEST(TextGenerationFeaturizerTest, FeaturizationWithPrompt) {
  std::vector<std::string> phrases = {
      R"({"prompt": "1 2", "target": "3 4 5 6 7 8 9"})"};

  std::vector<std::vector<std::vector<uint32_t>>> expected_indices = {
      // Prompt input
      {{1, 2}, {1, 2}, {1, 2}, {1, 2}, {1, 2}},
      //  LRC context input
      {{3}, {3, 4}, {3, 4, 5}, {3, 4, 5, 6}, {8}},
      // IRC context input
      {
          {3},
          expectedPairgrams({3, 4}),
          expectedPairgrams({3, 4, 5}),
          expectedPairgrams({4, 5, 6}),
          expectedPairgrams({8}),
      },
      // SRC context input
      {{0, 3}, {3, 4}, {4, 5}, {5, 6}, {0, 8}},
      // Labels
      {{4}, {5}, {6}, {7}, {9}}};

  checkDataFeaturization(phrases, expected_indices);
}

TEST(TextGenerationFeaturizerTest, FeaturizationWithPosition) {
  std::vector<std::string> phrases = {R"({"target": "1 2 3 4 5 6 7 8"})"};

  std::vector<std::vector<std::vector<uint32_t>>> expected_indices = {
      // Prompt input
      {{0}, {0}, {0}, {0}, {0}, {0}},
      //  LRC context input
      {{1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, {6}, {6, 7}},
      // IRC context input
      {
          {1},
          expectedPairgrams({1, 2}),
          expectedPairgrams({1, 2, 3}),
          expectedPairgrams({2, 3, 4}),
          {6},
          expectedPairgrams({6, 7}),
      },
      // SRC context input
      {{0, 1, 9}, {1, 2, 10}, {2, 3, 11}, {3, 4, 12}, {0, 6, 9}, {6, 7, 10}},
      // Labels
      {{2}, {3}, {4}, {5}, {7}, {8}}};

  checkDataFeaturization(phrases, expected_indices,
                         /* include_position= */ true);
}

TEST(TextGenerationFeaturizerTest, InferenceFeaturization) {
  std::vector<std::vector<std::vector<uint32_t>>> expected_indices = {
      // Prompt input
      {{0}},
      //  LRC context input
      {{2, 3, 4, 5}},
      // IRC context input
      {expectedPairgrams({3, 4, 5})},
      // SRC context input
      {{4, 5}}};

  checkInferenceFeaturization({}, {1, 2, 3, 4, 5}, expected_indices);
}

TEST(TextGenerationFeaturizerTest, InferenceFeaturizationWithPrompt) {
  std::vector<std::vector<std::vector<uint32_t>>> expected_indices = {
      // Prompt input
      {{7, 8, 9}},
      //  LRC context input
      {{1, 2}},
      // IRC context input
      {expectedPairgrams({1, 2})},
      // SRC context input
      {{1, 2}}};

  checkInferenceFeaturization({7, 8, 9}, {1, 2}, expected_indices);
}

TEST(TextClassifierFeaturizerTest, Featurization) {
  std::vector<std::vector<std::vector<uint32_t>>> expected_indices = {
      //  LRC context input
      {{1, 2}, {1, 2, 3}, {1, 2, 3, 4}},
      // IRC context input
      {expectedPairgrams({1, 2}), expectedPairgrams({1, 2, 3}),
       expectedPairgrams({2, 3, 4})},
      // SRC context input
      {{1, 2}, {2, 3}, {3, 4}},
      // Label
      {{0}, {1}, {2}}};

  TextClassificationFeaturizer featurizer(
      /* text_column= */ "text",
      /* label_column= */ "label",
      /* lrc_len= */ LRC_LEN,
      /* irc_len= */ IRC_LEN,
      /* src_len= */ SRC_LEN,
      /* vocab_size= */ VOCAB_SIZE,
      /* n_labels= */ 3, /* delimiter= */ ',',
      /* label_delimiter= */ std::nullopt, /* integer_labels= */ true,
      /* normalize_categories= */ true);

  featurizer.processHeader("text,label");

  auto data = featurizer.featurize({"1 2,0", "1 2 3,1", "1 2 3 4,2"});

  verifyGeneratedSamples(data, expected_indices);
}

}  // namespace thirdai::dataset::tests