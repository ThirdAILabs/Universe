#include "BlockTest.h"
#include <gtest/gtest.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Trend.h>
#include <dataset/src/utils/TimeUtils.h>
#include <sys/types.h>
#include <ctime>
#include <limits>
#include <memory>
#include <sstream>
#include <valarray>
#include <vector>

namespace thirdai::dataset {

class TrendBlockTests : public BlockTest {
 public:
  static StringMatrix makeTrivialSamples(uint32_t n_ids, uint32_t n_days,
                                         uint32_t day_offset,
                                         bool inc_by_id = false) {
    StringMatrix samples;
    for (uint32_t day = 0; day < n_days; day++) {
      for (uint32_t id = 0; id < n_ids; id++) {
        std::vector<std::string> sample;
        std::stringstream id_ss;
        id_ss << id;
        sample.push_back(id_ss.str());

        time_t timestamp = static_cast<time_t>(day + day_offset) *
                           SECONDS_IN_DAY;  // Add offset to prevent overflow
                                            // due to timezone differences.
        auto* tm = std::localtime(&timestamp);
        std::string timestamp_str;
        timestamp_str.resize(10);
        std::strftime(timestamp_str.data(), 10, "%Y-%m-%d", tm);
        sample.push_back(timestamp_str);

        auto count = inc_by_id ? id : 1;
        count = (day % 2) * count;
        std::stringstream count_ss;
        count_ss << count;
        sample.push_back(count_ss.str());

        samples.push_back(sample);
      }
    }
    return samples;
  }
};

TEST_F(TrendBlockTests, Trivial) {
  std::vector<std::shared_ptr<Block>> blocks{std::make_shared<TrendBlock>(
      /* has_count_col = */ true, /* id_col = */ 0,
      /* timestamp_col = */ 1, /* count_col = */ 2,
      /* horizon = */ 0, /* lookback = */ 2)};

  auto samples = makeTrivialSamples(/* n_ids = */ 1, /* n_days = */ 3650,
                                    /* day_offset = */ 365);
  auto vecs =
      makeSparseSegmentedVecs(samples, blocks, /* batch_interval = */ 10);

  size_t i = 0;
  for (auto& vec : vecs) {
    auto entries = vectorEntries(vec);
    ASSERT_EQ(entries.size(), 3);
    if (i == 0) {
      ASSERT_EQ(entries.at(0), 0);
      ASSERT_EQ(entries.at(1), 0);
      ASSERT_EQ(entries.at(2), 0);
    } else {
      ASSERT_EQ(entries.at(0), (i % 2) - 0.5);
      ASSERT_EQ(entries.at(1), ((i + 1) % 2) - 0.5);
      ASSERT_EQ(entries.at(2), 0.5);
    }
    i++;
  }
}

TEST_F(TrendBlockTests, CorrectCenteringAndNormalization) {
  
  size_t n_ids = 100;
  size_t lookback = 10;
  std::vector<std::shared_ptr<Block>> blocks{std::make_shared<TrendBlock>(
      /* has_count_col = */ true, /* id_col = */ 0,
      /* timestamp_col = */ 1, /* count_col = */ 2,
      /* horizon = */ 0, /* lookback = */ lookback)};

  auto samples =
      makeTrivialSamples(/* n_ids = */ n_ids, /* n_days = */ 365,
                         /* day_offset = */ 365, /* inc_by_id */ true);
  auto vecs = makeSparseSegmentedVecs(samples, blocks, 256);

  bool found_nonzero_mean = false;
  float delta = 1e-6;

  size_t i = 0;
  for (auto& vec : vecs) {
    auto entries = vectorEntries(vec);
    ASSERT_EQ(entries.size(), lookback + 1);
    auto mean = entries.at(lookback);
    if (i >= lookback * n_ids) {
      ASSERT_FLOAT_EQ(mean, static_cast<float>(i % 100) / 2);
    }
    float sum = 0;
    for (size_t i = 0; i < lookback; i++) {
      sum += entries[i];
    }
    ASSERT_LE(std::abs(sum - 0.0), delta);

    float expected_sum = mean * lookback;
    float actual_sum = 0;
    for (size_t i = 0; i < lookback; i++) {
      actual_sum += entries[i] * expected_sum + mean;
    }
    ASSERT_FLOAT_EQ(expected_sum, actual_sum);

    if (mean != 0.0) {
      found_nonzero_mean = true;
    }
    i++;
  }

  ASSERT_TRUE(found_nonzero_mean);
}

// TODO(Geordie): Should change lifetime to check number of samples too because
// otherwise if we have a window size of 1 and lag of zero we keep making new
// sketches. In fact, maybe only check number of samples instead of the dates.
// This allows for some neat things, like e.g. instead of having a critical
// section within the add vector segment method, we can do it before the
// parallel for loop.

}  // namespace thirdai::dataset