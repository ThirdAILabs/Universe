#include "gtest/gtest.h"
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/tests/BoltVectorTestUtils.h>
#include <hashing/src/HashUtils.h>
#include <dataset/src/featurizers/TextGenerationFeaturizer.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>

namespace thirdai::dataset::tests {

constexpr uint32_t VOCAB_SIZE = 8;

// Helper function to represent how pairgrams are encoded in the
// TextGenerationFeaturizer.
uint32_t pairgramHash(uint32_t lhs, uint32_t rhs) {
  uint32_t hash = hashing::combineHashes(lhs, rhs);
  hash = hash % (std::numeric_limits<uint32_t>::max() - VOCAB_SIZE);
  return hash + VOCAB_SIZE;
}

void verifyGeneratedSamples(
    const std::vector<std::string>& phrases,
    const std::vector<std::vector<std::vector<uint32_t>>>& expected_indices) {
  TextGenerationFeaturizer processor(/* lrc_len= */ 4, /* irc_len= */ 3,
                                     /* src_len= */ 2,
                                     /* vocab_size= */ VOCAB_SIZE);

  auto data = processor.featurize(phrases);
  ASSERT_EQ(data.size(), 4);

  for (uint32_t sample_id = 0; sample_id < expected_indices.size();
       sample_id++) {
    for (uint32_t input_id = 0;
         input_id < expected_indices.at(sample_id).size(); input_id++) {
      ASSERT_EQ(data.at(input_id).size(), expected_indices.size());

      ASSERT_EQ(data.at(input_id).at(sample_id).len,
                expected_indices.at(sample_id).at(input_id).size());

      for (uint32_t i = 0;
           i < expected_indices.at(sample_id).at(input_id).size(); i++) {
        ASSERT_EQ(data.at(input_id).at(sample_id).active_neurons[i],
                  expected_indices.at(sample_id).at(input_id).at(i));
        ASSERT_EQ(data.at(input_id).at(sample_id).activations[i], 1.0);
      }
    }
  }
}

TEST(TextGenerationFeaturizerTest, Featurization) {
  std::vector<std::string> phrases = {R"({"target": "1 2 3 4 5 6"})"};

  std::vector<std::vector<std::vector<uint32_t>>> expected_indices = {
      {{1}, {1}, {0, 1}, {2}},
      {{1, 2}, {1, 2, pairgramHash(1, 2)}, {1, 2}, {3}},
      {{1, 2, 3},
       {1, 2, 3, pairgramHash(1, 2), pairgramHash(1, 3), pairgramHash(2, 3)},
       {2, 3},
       {4}},
      {{1, 2, 3, 4},
       {2, 3, 4, pairgramHash(2, 3), pairgramHash(2, 4), pairgramHash(3, 4)},
       {3, 4},
       {5}},
      {{2, 3, 4, 5},
       {3, 4, 5, pairgramHash(3, 4), pairgramHash(3, 5), pairgramHash(4, 5)},
       {4, 5},
       {6}},
  };

  verifyGeneratedSamples(phrases, expected_indices);
}

TEST(TextGenerationFeaturizerTest, FeaturizationWithContext) {
  std::vector<std::string> phrases = {
      R"({"context": "1 2", "target": "3 4 5 6"})"};

  std::vector<std::vector<std::vector<uint32_t>>> expected_indices = {
      {{1, 2, 3},
       {1, 2, 3, pairgramHash(1, 2), pairgramHash(1, 3), pairgramHash(2, 3)},
       {2, 3},
       {4}},
      {{1, 2, 3, 4},
       {2, 3, 4, pairgramHash(2, 3), pairgramHash(2, 4), pairgramHash(3, 4)},
       {3, 4},
       {5}},
      {{2, 3, 4, 5},
       {3, 4, 5, pairgramHash(3, 4), pairgramHash(3, 5), pairgramHash(4, 5)},
       {4, 5},
       {6}},
  };

  verifyGeneratedSamples(phrases, expected_indices);
}

}  // namespace thirdai::dataset::tests