#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Trend.h>
#include <dataset/src/utils/TimeUtils.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <valarray>
#include <vector>

namespace thirdai::dataset {

class TrendBlockTests : public testing::Test {
 public:
  static constexpr float default_delta = 1E-7;

  static void assertFloatEq(float f1, float f2, float delta = default_delta) {
    ASSERT_LT(std::abs(f1 - f2), delta);
  }
};

std::unordered_map<uint32_t, float> vectorEntries(const BoltVector& vec) {
  std::unordered_map<uint32_t, float> features;
  for (uint32_t pos = 0; pos < vec.len; pos++) {
    uint32_t active_neuron = vec.isDense() ? pos : vec.active_neurons[pos];
    features[active_neuron] += vec.activations[pos];
  }
  return features;
}

TEST_F(TrendBlockTests, CorrectOutputWindowSizeOneNoGraph) {
  std::vector<std::string> input_rows = {
    "id_1,2022-01-01,0.3",
    "id_1,2022-01-02,0.5",
    "id_1,2022-01-08,0.1",
    "id_2,2022-01-08,0.45",
  };
  
  auto [vecs, _] = GenericBatchProcessor(
    /* input_blocks= */ {std::make_shared<TrendBlock>(
      /* has_count_col = */ true, /* id_col = */ 0,
      /* timestamp_col = */ 1, /* count_col = */ 2,
      /* lookahead = */ 0, /* lookback = */ 7, /* period = */ 7)},
    /* label_blocks= */ {}, /* has_header= */ true, /* delimiter= */ ',', /* parallel= */ false
  ).createBatch(input_rows);

  auto vec_0_entries = vectorEntries(vecs[0]);
  ASSERT_EQ(vec_0_entries.size(), 1);
  assertFloatEq(vec_0_entries.at(0), 0.3);
  for (auto [idx, val] : vec_0_entries) {
    std::cout << idx << " : " << val << std::endl;
  }
  
  auto vec_1_entries = vectorEntries(vecs[1]);
  ASSERT_EQ(vec_1_entries.size(), 1);
  assertFloatEq(vec_1_entries.at(0), 0.8);
  
  auto vec_2_entries = vectorEntries(vecs[2]);
  ASSERT_EQ(vec_2_entries.size(), 1);
  assertFloatEq(vec_2_entries.at(0), 0.1);
  
  auto vec_3_entries = vectorEntries(vecs[3]);
  ASSERT_EQ(vec_3_entries.size(), 1);
  assertFloatEq(vec_3_entries.at(0), 0.45);
}

TEST_F(TrendBlockTests, CorrectOutputLargerWindow) {
  
  std::vector<std::string> input_rows = {
      "id_1,2022-01-01,0.1",  // Trivial
      "id_1,2022-01-02,0.2",  // Check handle neighbor
      "id_1,2022-01-03,0.3",  // Check handle not in graph
      "id_1,2022-01-04,0.4",  // Check same period gets aggregated
  };

  auto [vecs, _] = GenericBatchProcessor(
    /* input_blocks= */ {std::make_shared<TrendBlock>(
      /* has_count_col = */ true, /* id_col = */ 0,
      /* timestamp_col = */ 1, /* count_col = */ 2,
      /* lookahead = */ 0, /* lookback = */ 4, /* period = */ 1)},
    /* label_blocks= */ {}, /* has_header= */ true, /* delimiter= */ ',', /* parallel= */ false
  ).createBatch(input_rows);

  auto vec_3_entries = vectorEntries(vecs[3]);
  ASSERT_EQ(vec_3_entries.size(), 4);
  float l2_norm = std::sqrt(2 * (0.15 * 0.15 + 0.05 * 0.05));
  assertFloatEq(vec_3_entries.at(0), 0.15 / l2_norm);
  assertFloatEq(vec_3_entries.at(1), 0.05 / l2_norm);
  assertFloatEq(vec_3_entries.at(2), -0.05 / l2_norm);
  assertFloatEq(vec_3_entries.at(3), -0.15 / l2_norm);
}

TEST_F(TrendBlockTests, PrepareBatchWorks) {
  /*
    Trend block's prepareBatch has a mechanism to delete outdated counts 
    (older than lookback days + lookahead days)
    Trend block internally stores a "recent" sketch and an "old" sketch
    When trend block sees a new batch, it deletes the "old" sketch, 
    converts the "recent" sketch to an "old" sketch, and create a new 
    "recent" sketch.
    Thus, a count gets deleted permanently if Trend block sees a timestamp
    that is (lookback days + lookahead days) ahead of the current timestamp
    2 batches in a row.
  */
  
  std::vector<std::string> batch_0(10000);
  std::fill(batch_0.begin(), batch_0.end(), "id_2,2022-01-01,5");

  std::vector<std::string> batch_1(10000);
  std::fill(batch_1.begin(), batch_1.end(), "id_1,2022-01-01,5");
  
  std::vector<std::string> batch_2(10000);
  std::fill(batch_2.begin(), batch_2.end(), "id_1,2022-01-07,5");
  
  std::vector<std::string> batch_3(10000);
  batch_3[0] = "id_1,2022-01-13,0.1";
  std::fill(batch_3.begin() + 1, batch_3.end(), "id_1,2022-01-01,5");

  GenericBatchProcessor processor(
    /* input_blocks= */ {std::make_shared<TrendBlock>(
      /* has_count_col = */ true, /* id_col = */ 0,
      /* timestamp_col = */ 1, /* count_col = */ 2,
      /* lookahead = */ 0, /* lookback = */ 3, /* period = */ 3)},
    /* label_blocks= */ {}, /* has_header= */ true, /* delimiter= */ ',', /* parallel= */ false
  );
  
  processor.createBatch(batch_0);
  
  auto [vecs_1, _1] = processor.createBatch(batch_1);
  auto vec_1_entries = vectorEntries(vecs_1[0]);
  ASSERT_EQ(vec_1_entries.size(), 1);
  ASSERT_EQ(vec_1_entries.at(0), 5);
  
  auto [vecs_2, _2] = processor.createBatch(batch_1);
  auto vec_2_entries = vectorEntries(vecs_2[0]);
  ASSERT_EQ(vec_2_entries.size(), 1);
  ASSERT_EQ(vec_2_entries.at(0), 50005);
  
  processor.createBatch(batch_2);

  auto [vecs_3, _3] = processor.createBatch(batch_3);
  auto vec_3_entries = vectorEntries(vecs_3[1]);
  ASSERT_EQ(vec_3_entries.size(), 1);
  ASSERT_EQ(vec_3_entries.at(0), 5); // As opposed to 15
}

TEST_F(TrendBlockTests, TryExplain) {
  TrendBlock block(
      /* has_count_col = */ true, /* id_col = */ 0,
      /* timestamp_col = */ 1, /* count_col = */ 2,
      /* lookahead = */ 0, /* lookback = */ 12, /* period = */ 3);
  
  std::string zero = "0";
  std::string timestamp = "2022-02-02";
  std::string_view zero_view = {zero.data(), zero.length()};
  std::string_view timestamp_view = {timestamp.data(), timestamp.length()};
  std::vector<std::string_view> row = {zero_view, timestamp_view, zero_view};
  
  std::unordered_map<uint32_t, std::string> num_to_name = {
    {0, "user"},
    {1, "timestamp"},
    {2, "count"},
  };

  auto a = block.explainFeature(/* index_within_block= */ 1, num_to_name, row);
  std::cout << a.column_name << std::endl;
  std::cout << a.input_key << std::endl;
}

}  // namespace thirdai::dataset