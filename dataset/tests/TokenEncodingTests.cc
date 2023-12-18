#include <gtest/gtest.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <utils/text/StringManipulation.h>
#include <unordered_set>

namespace thirdai::dataset {

TEST(TokenEncodingTest, VerifyNumberOfNGrams) {
  std::string sentence = "This is a sentence with many words.";
  uint32_t num_words = 7;

  for (uint32_t n = 1; n < 8; n++) {
    auto n_gram_tokens = token_encoding::ngrams(sentence, /* n= */ n);

    uint32_t expected_num_ngrams;
    // - ngrams always includes unigrams regardless of the N value
    // - if N is too large we can't make an NGram and will only have unigrams
    if (n == 1 || n > num_words) {
      expected_num_ngrams = num_words;
    } else if (n <= num_words) {
      expected_num_ngrams = num_words + (num_words - n + 1);
    }

    ASSERT_EQ(n_gram_tokens.size(), expected_num_ngrams);
  }
}

TEST(TokenEncodingTest, TestSumRepeatedIndices) {
  std::vector<uint32_t> indices{1, 2, 3, 3, 1, 1};

  auto index_value_pairs = token_encoding::sumRepeatedIndices(indices);

  ASSERT_EQ(index_value_pairs[0].first, 1);
  ASSERT_EQ(index_value_pairs[0].second, 3.0);

  ASSERT_EQ(index_value_pairs[1].first, 2);
  ASSERT_EQ(index_value_pairs[1].second, 1.0);

  ASSERT_EQ(index_value_pairs[2].first, 3);
  ASSERT_EQ(index_value_pairs[2].second, 2.0);
}

TEST(TokenEncodingTest, TestUnigramPreservingPairgrams) {
  std::vector<uint32_t> tokens = {1, 2, 3, 4, 5};

  auto pairgrams = token_encoding::unigramPreservingPairgrams(
      tokens.data(), tokens.size(), 10);

  // Check unigrams are preserved.
  for (uint32_t i = 0; i < tokens.size(); i++) {
    ASSERT_EQ(tokens[i], pairgrams[i]);
  }

  // Check that we have the expected number of pairgrams.
  std::unordered_set<uint32_t> unique_pairgrams;
  for (uint32_t i = tokens.size(); i < pairgrams.size(); i++) {
    // Check that pairgrams are in the correct range.
    ASSERT_GE(pairgrams[i], 10);
    unique_pairgrams.insert(pairgrams[i]);
  }

  ASSERT_EQ(unique_pairgrams.size(), tokens.size() * (tokens.size() - 1) / 2);
}

}  // namespace thirdai::dataset