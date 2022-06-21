#include "BlockTest.h"
#include <gtest/gtest.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/CountHistory.h>
#include <dataset/src/encodings/count_history/DynamicCounts.h>
#include <dataset/src/utils/TimeUtils.h>
#include <sys/types.h>
#include <ctime>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace thirdai::dataset {

class CountHistoryBlockTests : public BlockTest {
 public:
  static StringMatrix makeTrivialSamples(uint32_t n_ids, uint32_t n_days, uint32_t day_offset, bool inc_by_id=false) {
    StringMatrix samples;
    for (uint32_t day = 0; day < n_days; day++) {
      for (uint32_t id = 0; id < n_ids; id++) {
        std::vector<std::string> sample;
        std::stringstream id_ss;
        id_ss << id;
        sample.push_back(id_ss.str());
        
        time_t timestamp = static_cast<time_t>(day + day_offset) * SECONDS_IN_DAY; // Add offset to prevent overflow due to timezone differences.
        auto* tm = std::localtime(&timestamp);
        std::string timestamp_str;
        timestamp_str.resize(10);
        std::strftime(timestamp_str.data(), 10, "%Y-%m-%d", tm);
        sample.push_back(timestamp_str);

        if (inc_by_id) {
          sample.push_back(id_ss.str());
        }

        samples.push_back(sample);
      }
    }
    return samples;
  }
  static StringMatrix genNearNeighbours(uint32_t n, uint32_t k) {
    StringMatrix near_neighbours;
    for(uint32_t i=0;i<n;i++) {
      std::vector<std::string> sample;
      sample.push_back(std::to_string(i));
      for (uint32_t j=0;j<k;j++) {
        sample.push_back(std::to_string((i+1+j)%n));
      }
      near_neighbours.push_back(sample);
    }
    return near_neighbours;
  }
};

TEST_F(CountHistoryBlockTests, Trivial) {
  std::vector<Window> windows{Window(/* lag = */ 0, /* size = */ 1)};
  DynamicCountsConfig config(/* max_range = */ 1, /* n_rows = */ 5, /* range_pow = */ 10);
  std::vector<std::shared_ptr<Block>> blocks{std::make_shared<CountHistoryBlock>(/* has_count_col = */ false, /* id_col = */ 0, /* timestamp_col = */ 1, /* count_col = */ 0, windows, config)};

  auto samples = makeTrivialSamples(/* n_ids = */ 1, /* n_days = */ 3650, /* day_offset = */ 365);
  auto vecs = makeSparseSegmentedVecs(samples, blocks);

  for (auto& vec : vecs) {
    auto entries = vectorEntries(vec);
    ASSERT_EQ(entries.size(), 1);
    ASSERT_EQ(entries[0], 1.0);
  }
}

TEST_F(CountHistoryBlockTests, TrivialManyUsers) {
  std::vector<Window> windows{Window(/* lag = */ 0, /* size = */ 1)};
  DynamicCountsConfig config(/* max_range = */ 1, /* n_rows = */ 5, /* range_pow = */ 20);
  std::vector<std::shared_ptr<Block>> blocks{std::make_shared<CountHistoryBlock>(/* has_count_col = */ false, /* id_col = */ 0, /* timestamp_col = */ 1, /* count_col = */ 0, windows, config)};

  auto samples = makeTrivialSamples(/* n_ids = */ 500, /* n_days = */ 365, /* day_offset = */ 365);
  auto vecs = makeSparseSegmentedVecs(samples, blocks);

  for (auto& vec : vecs) {
    auto entries = vectorEntries(vec);
    ASSERT_EQ(entries.size(), 1);
    ASSERT_EQ(entries[0], 1.0);
  }
}

TEST_F(CountHistoryBlockTests, ManyWindowsManyUsers) {
  std::vector<Window> windows{
    Window(/* lag = */ 0, /* size = */ 1), 
    Window(/* lag = */ 1, /* size = */ 2), 
    Window(/* lag = */ 3, /* size = */ 3), 
    Window(/* lag = */ 6, /* size = */ 4)};
  DynamicCountsConfig config(/* max_range = */ 1, /* n_rows = */ 5, /* range_pow = */ 20);
  std::vector<std::shared_ptr<Block>> blocks{std::make_shared<CountHistoryBlock>(/* has_count_col = */ false, /* id_col = */ 0, /* timestamp_col = */ 1, /* count_col = */ 0, windows, config)};

  uint32_t n_ids = 500;
  auto samples = makeTrivialSamples(n_ids, /* n_days = */ 365, /* day_offset = */ 365);
  auto vecs = makeSparseSegmentedVecs(samples, blocks);


  for (size_t i = 0; i < vecs.size(); i++) {
    size_t day = i / n_ids;
    auto entries = vectorEntries(vecs[i]);
    ASSERT_EQ(entries.size(), 4);
    ASSERT_EQ(entries[0], 1.0);
    ASSERT_EQ(entries[1], std::min(2.0, static_cast<double>(day)));
    ASSERT_EQ(entries[2], std::min(3.0, static_cast<double>(std::max(static_cast<int>(day - 2), 0))));
    ASSERT_EQ(entries[3], std::min(4.0, static_cast<double>(std::max(static_cast<int>(day - 5), 0))));
    
  }
}

TEST_F(CountHistoryBlockTests, SingleUserDifferentFrequencies) {
  StringMatrix samples;
  uint32_t day = 0;
  for (uint32_t i = 0; i < 1; i++) {
    day += i;
    std::vector<std::string> sample;
    sample.push_back("0");
    
    time_t timestamp = static_cast<time_t>(day + 365) * SECONDS_IN_DAY; // Add offset to prevent overflow due to timezone differences.
    auto* tm = std::localtime(&timestamp);
    std::string timestamp_str;
    timestamp_str.resize(10);
    std::strftime(timestamp_str.data(), 10, "%Y-%m-%d", tm);
    sample.push_back(timestamp_str);
    
    samples.push_back(sample);
  }

  std::vector<Window> windows{Window(/* lag = */ 0, /* size = */ 10)};
  DynamicCountsConfig config(/* max_range = */ 1, /* n_rows = */ 5, /* range_pow = */ 20);
  std::vector<std::shared_ptr<Block>> blocks{std::make_shared<CountHistoryBlock>(/* has_count_col = */ false, /* id_col = */ 0, /* timestamp_col = */ 1, /* count_col = */ 0, windows, config)};

  std::vector<float> expectations{
    1.0, // day 0
    2.0, // day 1
    3.0, // day 3
    4.0, // day 6
    4.0, // day 10
    3.0, // day 15
    2.0, // day 21
    2.0, // day 28
    2.0, // day 36
    2.0, // day 45
  };

  auto vecs = makeSparseSegmentedVecs(samples, blocks);


  for (size_t i = 0; i < vecs.size(); i++) {
    auto entries = vectorEntries(vecs[i]);
    ASSERT_EQ(entries.size(), 1);
    ASSERT_EQ(entries[0], expectations[i]);
  }
}

TEST_F(CountHistoryBlockTests, ManyUsersManyIncrements) {
  std::vector<Window> windows{Window(/* lag = */ 0, /* size = */ 1)};
  DynamicCountsConfig config(/* max_range = */ 1, /* n_rows = */ 5, /* range_pow = */ 20);
  std::vector<std::shared_ptr<Block>> blocks{std::make_shared<CountHistoryBlock>(/* has_count_col = */ true, /* id_col = */ 0, /* timestamp_col = */ 1, /* count_col = */ 2, windows, config)};

  uint32_t n_ids = 500;
  auto samples = makeTrivialSamples(n_ids, /* n_days = */ 365, /* day_offset = */ 365, /* inc_by_id = */ true);
  auto vecs = makeSparseSegmentedVecs(samples, blocks);

  for (uint32_t i = 0; i < vecs.size(); i++) {
    uint32_t id = i % n_ids;
    float expectation = static_cast<float>(id);
    auto entries = vectorEntries(vecs[i]);
    ASSERT_EQ(entries.size(), 1);
    ASSERT_EQ(entries[0], expectation);
  }
}

TEST_F(CountHistoryBlockTests, ErrorDoesNotGrowOverTime) {
  uint32_t n_ids = 500;
  auto samples = makeTrivialSamples(n_ids, /* n_days = */ 365, /* day_offset = */ 365);
  std::vector<Window> windows{Window(/* lag = */ 0, /* size = */ 30)};
  DynamicCountsConfig config(/* max_range = */ 1, /* n_rows = */ 5, /* range_pow = */ 14);
  std::vector<std::shared_ptr<Block>> blocks{std::make_shared<CountHistoryBlock>(/* has_count_col = */ false, /* id_col = */ 0, /* timestamp_col = */ 1, /* count_col = */ 0, windows, config)};
  auto vecs = makeSparseSegmentedVecs(samples, blocks);
  // Intentionally add too many, and check that error does not increase after first month.
  
  float max = 0;
  for (uint32_t i = 0; i < vecs.size() / 2; i++) {
    auto entries = vectorEntries(vecs[i]);
    max = std::max(max, entries[0]);
  }

  for (uint32_t i = vecs.size() / 2; i < vecs.size(); i++) {
    auto entries = vectorEntries(vecs[i]);
    ASSERT_LE(entries[0], max * 1.1);
  }

}

TEST_F(CountHistoryBlockTests, NearNeighbours) {
  std::vector<Window> windows{Window(/* lag = */ 0, /* size = */ 2)};
  DynamicCountsConfig config(/* max_range = */ 1, /* n_rows = */ 5, /* range_pow = */ 10);
  StringMatrix near_neighbours = genNearNeighbours(3,2);
  std::vector<std::shared_ptr<Block>> blocks{std::make_shared<CountHistoryBlock>(/* has_count_col = */ false, /* id_col = */ 0, /* timestamp_col = */ 1, /* count_col = */ 0, windows, config,true,near_neighbours)};

  auto samples = makeTrivialSamples(/* n_ids = */ 3, /* n_days = */ 2, /* day_offset = */ 365);
  auto vecs = makeSparseSegmentedVecs(samples, blocks);
  std::vector<uint32_t> num(3,0);
  for (uint32_t i =0;i<vecs.size();i++) {
    auto entries = vectorEntries(vecs[i]);
    uint32_t a,b,c;
    switch(i%3) {
      case 0: {
        num[0]++;
        a = num[0];b=num[1];c = num[2];
        break;
      }
      case 1: {
        num[1]++;
        a = num[1];b=num[2];c = num[0];
        break;
      }
      case 2: {
        num[2]++;
        a = num[2];b=num[0];c = num[1];
        break;
      }
    }
    ASSERT_EQ(entries.size(), 3);
    ASSERT_EQ(entries[0], a);
    ASSERT_EQ(entries[1], b);
    ASSERT_EQ(entries[2], c);
  }
}


// TODO(Geordie): Should change lifetime to check number of samples too because otherwise if we have a window size of 1 and lag of zero we keep making new sketches.
// In fact, maybe only check number of samples instead of the dates. This allows for some neat things, like e.g. instead of having a critical section within the 
// add vector segment method, we can do it before the parallel for loop.

} // namespace thirdai::dataset