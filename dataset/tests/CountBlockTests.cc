#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/blocks/BlockList.h>
#include <dataset/src/blocks/Count.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <utils/StringManipulation.h>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::dataset::tests {

static uint32_t MAX_LEN = 10;
static uint32_t N_SAMPLES = 10;
static const char* DELIM = " ";

TabularFeaturizer featurizer() {
  return TabularFeaturizer(
      /* block_lists= */ {BlockList({dataset::CountBlock::make(
          /* column= */ 0, /* delimiter= */ DELIM[0],
          /* ceiling= */ MAX_LEN + 1)})},
      /* has_header= */ false);
}

std::vector<std::vector<std::string>> unjoinedSamples(uint32_t min_len,
                                                      uint32_t max_len) {
  std::random_device rd;   // obtain a random number from hardware
  std::mt19937 gen(rd());  // seed the generator
  std::uniform_int_distribution<> distr(min_len, max_len);  // define the range

  std::vector<std::vector<std::string>> samples(N_SAMPLES);
  for (auto& sample : samples) {
    uint32_t len = distr(gen);
    sample = std::vector<std::string>(len, "a");
  }

  return samples;
}

std::vector<std::string> join(
    const std::vector<std::vector<std::string>>& unjoined_samples) {
  std::vector<std::string> joined;
  joined.reserve(unjoined_samples.size());
  for (const auto& sample : unjoined_samples) {
    joined.push_back(text::join(sample, /* delimiter= */ DELIM));
  }
  return joined;
}

TEST(CountBlockTests, CorrectCounts) {
  auto unjoined_samples =
      unjoinedSamples(/* min_len= */ 0, /* max_len= */ MAX_LEN);
  auto samples = join(unjoined_samples);
  auto feat = featurizer();
  auto batches = feat.featurize(samples);
  auto vectors = batches.front();

  ASSERT_EQ(vectors.size(), unjoined_samples.size());

  for (uint32_t i = 0; i < vectors.size(); i++) {
    ASSERT_EQ(vectors[i].len, 1);
    ASSERT_EQ(vectors[i].active_neurons[0], unjoined_samples[i].size());
    ASSERT_EQ(vectors[i].activations[0], 1.0);
  }
}

TEST(CountBlockTests, ThrowsWhenSequenceIsTooLong) {
  auto unjoined_samples =
      unjoinedSamples(/* min_len= */ MAX_LEN + 1, /* max_len= */ 2 * MAX_LEN);
  auto samples = join(unjoined_samples);
  auto feat = featurizer();

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      feat.featurize(samples), std::invalid_argument);
}

}  // namespace thirdai::dataset::tests