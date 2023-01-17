#include <gtest/gtest.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <string_view>

namespace thirdai::dataset {

TEST(TokenEncodingTest, VerifyNumberOfNGrams) {
  std::string_view sentence = "This is a sentence with many words.";
  uint32_t num_words = 7;

  for (uint32_t n = 1; n < 8; n++) {
    auto n_gram_tokens = TokenEncoding::computeNGrams(sentence, /* n= */ n);

    uint32_t expected_num_ngrams;
    // - computeNGrams always includes unigrams regardless of the N value
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

  auto index_value_pairs = TokenEncoding::sumRepeatedIndices(indices);

  ASSERT_EQ(index_value_pairs[0].first, 1);
  ASSERT_EQ(index_value_pairs[0].second, 3.0);

  ASSERT_EQ(index_value_pairs[1].first, 1);
  ASSERT_EQ(index_value_pairs[1].second, 1.0);

  ASSERT_EQ(index_value_pairs[2].first, 1);
  ASSERT_EQ(index_value_pairs[2].second, 2.0);
}

}  // namespace thirdai::dataset