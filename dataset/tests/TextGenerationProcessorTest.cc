#include "gtest/gtest.h"
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/tests/BoltVectorTestUtils.h>
#include <hashing/src/HashUtils.h>
#include <dataset/src/text_generation/TextGenerationProcessor.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <unordered_set>

namespace thirdai::dataset::tests {

std::vector<std::string> convertPhrasesToJson(
    const std::vector<std::string>& phrases) {
  std::vector<std::string> json;

  for (const auto& phrase : phrases) {
    // Converts to {"text":"<phrase>"}
    std::string formatted_phrase = R"({"text":")" + phrase + R"("})";
    json.emplace_back(std::move(formatted_phrase));
  }

  return json;
}

void verifyPairgramIntersection(const BoltVector& a, const BoltVector& b,
                                uint32_t expected_intersection_size) {
  std::unordered_set<uint32_t> a_set(a.active_neurons,
                                     a.active_neurons + a.len);

  // Use an set here to make sure that we don't double count elements of b which
  // match to the same element of a.
  std::unordered_set<uint32_t> intersection;
  for (const auto* an = b.active_neurons; an != b.active_neurons + b.len;
       ++an) {
    if (a_set.count(*an)) {
      intersection.insert(*an);
    }
  }

  ASSERT_EQ(intersection.size(), expected_intersection_size);
}

TEST(TextGenerationProcessorTest, Featurization) {
  const uint32_t seq_len = 4;
  const uint32_t input_dim = 2000000;
  const uint32_t output_dim = 1000000;

  std::vector<std::string> phrases = {
      "aa bb cc dd ee ff gg", "mm nn oo pp qq rr ss tt", "uu vv ww xx yy zz"};

  std::vector<std::vector<uint32_t>> next_words_per_phrase = {
      token_encoding::unigrams("ee ff gg"),
      token_encoding::unigrams("qq rr ss tt"),
      token_encoding::unigrams("yy zz"),
  };

  TextGenerationProcessor processor(seq_len, input_dim, output_dim);

  auto [vectors, labels] = processor.featurize(convertPhrasesToJson(phrases));

  ASSERT_EQ(vectors.size(), 9);
  ASSERT_EQ(labels.size(), 9);

  /**
   * The returned `vectors` and `labels` are flattened lists of the generated
   * BoltVectors, however we iterate over them using the lengths of `next_words`
   * in order to group the samples from the same pharse so we can check the
   * intersection of the paigrams in the input vector.
   */
  uint32_t seq_index = 0;
  for (const auto& next_words : next_words_per_phrase) {
    for (uint32_t i = 0; i < next_words.size(); i++) {
      // Input vector length should be the number of pairgrams.
      EXPECT_EQ(vectors[seq_index + i].len, (seq_len + 1) * seq_len / 2);

      uint32_t next_word_unigram = next_words[i];

      BoltVector expected_label = BoltVector::makeSparseVector(
          {next_word_unigram % output_dim,
           hashing::simpleIntegerHash(next_word_unigram) % output_dim},
          {1.0, 1.0});

      // Labels should be the unigram of the label and a simple hash of that
      // unigram.
      thirdai::tests::BoltVectorTestUtils::assertBoltVectorsAreEqual(
          labels[seq_index + i], expected_label);

      // The activations of the input should all be 1.0.
      for (uint32_t j = 0; j < vectors[seq_index + i].len; i++) {
        ASSERT_EQ(vectors[seq_index + i].activations[j], 1.0);
      }
    }

    // Consecutive parigrams should have an intersection of size
    // (seq_len-1) * seq_len / 2
    for (uint32_t i = 0; i < next_words.size() - 1; i++) {
      verifyPairgramIntersection(vectors[seq_index + i],
                                 vectors[seq_index + i + 1],
                                 (seq_len - 1) * seq_len / 2);
    }

    seq_index += next_words.size();
  }
}

}  // namespace thirdai::dataset::tests