#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Trend.h>
#include <dataset/src/utils/TimeUtils.h>
#include <sys/types.h>
#include <cmath>
#include <ctime>
#include <limits>
#include <memory>
#include <sstream>
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
    features[vec.active_neurons[pos]] += vec.activations[pos];
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
    /* label_blocks= */ {}
  ).createBatch(input_rows);
  
  auto vec_0_entries = vectorEntries(vecs[0]);
  ASSERT_EQ(vec_0_entries.size(), 1);
  std::cout << vec_0_entries.at(0);
  assertFloatEq(vec_0_entries.at(0), 0.3);

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
  std::vector<std::shared_ptr<Block>> blocks{std::make_shared<TrendBlock>(
      /* has_count_col = */ true, /* id_col = */ 0,
      /* timestamp_col = */ 1, /* count_col = */ 2,
      /* lookahead = */ 0, /* lookback = */ 4, /* period = */ 1)};

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
    /* label_blocks= */ {}
  ).createBatch(input_rows);

  auto vec_3_entries = vectorEntries(vecs[3]);
  ASSERT_EQ(vec_3_entries.size(), 4);
  float l2_norm = std::sqrt(2 * (0.15 * 0.15 + 0.05 * 0.05));
  assertFloatEq(vec_3_entries.at(0), 0.15 / l2_norm);
  assertFloatEq(vec_3_entries.at(1), 0.05 / l2_norm);
  assertFloatEq(vec_3_entries.at(2), -0.05 / l2_norm);
  assertFloatEq(vec_3_entries.at(3), -0.15 / l2_norm);
}

}  // namespace thirdai::dataset