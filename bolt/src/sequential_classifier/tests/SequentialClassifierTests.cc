#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>
#include <bolt/src/sequential_classifier/SequentialClassifier.h>

namespace thirdai::bolt {
class SequentialClassifierTests : public testing::Test {
 public:

static constexpr const char* MOCK_FILE = "mock.csv";

static void writeMockFile() {
  std::ofstream out(MOCK_FILE);
  out << "user_id,item_id,timestamp,rating" << std::endl;
  for (size_t i = 0; i < 1000; i++) {
    time_t timestamp = static_cast<time_t>(i * 86400);  // Add offset to prevent overflow
                                            // due to timezone differences.
    auto* tm = std::localtime(&timestamp);
    std::string timestamp_str;
    timestamp_str.resize(10);
    std::strftime(timestamp_str.data(), 10, "%Y-%m-%d", tm);
    
    out << "0,0," << timestamp_str << ",5";
  }
  out.close();
}

static void removeMockFile() {
  remove(MOCK_FILE);
}

static std::unordered_map<std::string, std::string> buildSchema() {
  std::unordered_map<std::string, std::string> schema;
  schema["user"] = "user_id";
  schema["item"] = "item_id";
  schema["timestamp"] = "timestamp";
  schema["target"] = "rating";
  return schema;
}

// std::string task, size_t horizon, size_t n_items,
//                              size_t n_users = 0, size_t n_item_categories = 0,
                             

static SequentialClassifierConfig buildConfig() {
  return {/* task = */ "regression", /* horizon = */ 0, /* n_items = */ 1, /*n_users = */ 1};
}

static size_t userTemporalBlockUseCount(SequentialClassifier& seq) {
  return seq._user_trend_block->useCount();
}

static size_t itemTemporalBlockUseCount(SequentialClassifier& seq) {
  return seq._item_trend_block->useCount();
}

};

TEST_F(SequentialClassifierTests, ReuseTemporalBlocksForPrediction) {
  writeMockFile();
  SequentialClassifier seq(buildSchema(), buildConfig());
  seq.train(MOCK_FILE);
  ASSERT_EQ(userTemporalBlockUseCount(seq), 1);
  ASSERT_EQ(itemTemporalBlockUseCount(seq), 1);
  seq.predict(MOCK_FILE);
  ASSERT_EQ(userTemporalBlockUseCount(seq), 2);
  ASSERT_EQ(itemTemporalBlockUseCount(seq), 2);
  removeMockFile();
}
TEST_F(SequentialClassifierTests, NewTemporalBlocksForTraining) {
  writeMockFile();
  SequentialClassifier seq(buildSchema(), buildConfig());
  seq.train(MOCK_FILE);
  ASSERT_EQ(userTemporalBlockUseCount(seq), 1);
  ASSERT_EQ(itemTemporalBlockUseCount(seq), 1);
  seq.train(MOCK_FILE);
  ASSERT_EQ(userTemporalBlockUseCount(seq), 1);
  ASSERT_EQ(itemTemporalBlockUseCount(seq), 1);
  seq.predict(MOCK_FILE);
  ASSERT_EQ(userTemporalBlockUseCount(seq), 2);
  ASSERT_EQ(itemTemporalBlockUseCount(seq), 2);
  seq.train(MOCK_FILE);
  ASSERT_EQ(userTemporalBlockUseCount(seq), 1);
  ASSERT_EQ(itemTemporalBlockUseCount(seq), 1);
  removeMockFile();
}
TEST_F(SequentialClassifierTests, SameModelForTraining) {
  writeMockFile();
  SequentialClassifier seq(buildSchema(), buildConfig());
  seq.train(MOCK_FILE);
  float rmse_after_train_once = seq.predict(MOCK_FILE);
  seq.train(MOCK_FILE);
  float rmse_after_train_twice = seq.predict(MOCK_FILE);
  ASSERT_LT(rmse_after_train_twice, rmse_after_train_once);
  removeMockFile();
}
} // namespace thirdai::bolt