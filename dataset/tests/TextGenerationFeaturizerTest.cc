#include "gtest/gtest.h"
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/tests/BoltVectorTestUtils.h>
#include <hashing/src/HashUtils.h>
#include <dataset/src/featurizers/TextGenerationFeaturizer.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <sstream>
#include <string>
#include <unordered_set>

namespace thirdai::dataset::tests {

std::string ascendingIntegerString(uint32_t start, uint32_t n) {
  std::string str;
  for (uint32_t i = start; i < start + n; i++) {
    str += std::to_string(i) + ' ';
  }
  str.pop_back();
  return str;
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

TEST(TextGenerationFeaturizerTest, Featurization) {
  const uint32_t seq_len = 4;
  const uint32_t output_dim = 1000000;

  std::vector<std::string> phrases = {ascendingIntegerString(0, 7),
                                      ascendingIntegerString(7, 8),
                                      ascendingIntegerString(15, 6)};

  std::vector<std::vector<uint32_t>> expected_labels = {
      {4, 5, 6},
      {11, 12, 13, 14},
      {19, 20},
  };

  TextGenerationFeaturizer processor(seq_len, output_dim);

  auto data = processor.featurize(phrases);
  ASSERT_EQ(data.size(), 2);
  ASSERT_EQ(data.at(0).size(), 9);
  ASSERT_EQ(data.at(1).size(), 9);

  /**
   * The returned data contains flattened lists of the generated BoltVectors,
   * however we iterate over them using the lengths of `expected_labels` in
   * order to group the samples from the same pharse so we can check the
   * intersection of the paigrams in the input vector.
   */
  uint32_t seq_index = 0;
  for (const auto& labels : expected_labels) {
    for (uint32_t i = 0; i < labels.size(); i++) {
      // Input vector length should be the number of pairgrams.
      EXPECT_EQ(data.at(0).at(seq_index + i).len, (seq_len + 1) * seq_len / 2);

      BoltVector expected_label =
          BoltVector::singleElementSparseVector(labels.at(i));

      // Labels should be the unigram of the label and a simple hash of that
      // unigram.
      thirdai::tests::BoltVectorTestUtils::assertBoltVectorsAreEqual(
          data.at(1).at(seq_index + i), expected_label);

      // The activations of the input should all be 1.0.
      for (uint32_t j = 0; j < data.at(0).at(seq_index + i).len; j++) {
        ASSERT_EQ(data.at(0).at(seq_index + i).activations[j], 1.0);
      }
    }

    // Consecutive parigrams should have an intersection of size
    // (seq_len-1) * seq_len / 2
    for (uint32_t i = 0; i < labels.size() - 1; i++) {
      verifyPairgramIntersection(data.at(0).at(seq_index + i),
                                 data.at(0).at(seq_index + i + 1),
                                 (seq_len - 1) * seq_len / 2);
    }

    seq_index += labels.size();
  }
}

}  // namespace thirdai::dataset::tests