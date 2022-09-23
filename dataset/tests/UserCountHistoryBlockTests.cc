#include <gtest/gtest.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/UserCountHistory.h>
#include <dataset/src/utils/CountHistoryMap.h>
#include <cmath>

namespace thirdai::dataset {

static BoltBatch processBatch(BlockPtr block,
                              const std::vector<std::string>& input_rows) {
  GenericBatchProcessor processor(
      /* input_blocks= */ {block}, /* label_blocks= */ {},
      /* has_header= */ false, /* delimiter= */ ',', /* parallel= */ false);
  auto [batch, _] = processor.createBatch(input_rows);
  return std::move(batch);
}

TEST(UserCountHistoryBlockTest, ExplanationWorks) {
  auto count_history = CountHistoryMap::make(/* lookahead_periods= */ 1,
                                             /* lookback_periods= */ 5);

  auto block =
      UserCountHistoryBlock::make(/* user_col= */ 0, /* count_col= */ 1,
                                  /* timestamp_col= */ 2, count_history);
  
  std::vector<std::string> input_rows = {
      {"user,0,2022-02-02"}, {"user,1,2022-02-03"}, {"user,2,2022-02-04"},
      {"user,3,2022-02-05"}, {"user,4,2022-02-06"}, {"user,5,2022-02-07"},
  };

  auto batch = processBatch(block, input_rows);

  auto input_row = ProcessorUtils::parseCsvRow(input_rows[5], ',');

  auto explanation_0 = block->explainIndex(/* index_within_block= */ 0, input_row);
  ASSERT_EQ(explanation_0.column_number, 1); // Count is in column 2 of input row
  ASSERT_EQ(explanation_0.keyword, "between 2022-02-02 and 2022-02-03 value is lower than usual");
  
  auto explanation_2 = block->explainIndex(/* index_within_block= */ 2, input_row);
  ASSERT_EQ(explanation_2.column_number, 1); // Count is in column 2 of input row
  ASSERT_EQ(explanation_2.keyword, "between 2022-02-04 and 2022-02-05 value is same as usual");
  
  auto explanation_4 = block->explainIndex(/* index_within_block= */ 4, input_row);
  ASSERT_EQ(explanation_4.column_number, 1); // Count is in column 2 of input row
  ASSERT_EQ(explanation_4.keyword, "between 2022-02-06 and 2022-02-07 value is higher than usual");
}

TEST(UserCountHistoryBlockTest, NoNormalizeWhenLookbackPeriodsEqualsOne) {
  auto count_history = CountHistoryMap::make(/* lookahead_periods= */ 1,
                                             /* lookback_periods= */ 1);

  auto block =
      UserCountHistoryBlock::make(/* user_col= */ 0, /* count_col= */ 1,
                                  /* timestamp_col= */ 2, count_history);

  std::vector<std::string> input_rows = {
      {"user,33,2022-02-02"},
      {"user,0,2022-02-03"},
  };

  auto batch = processBatch(block, input_rows);

  auto last_vector = batch[1];

  ASSERT_TRUE(last_vector.isDense());
  ASSERT_EQ(last_vector.len, 1);
  ASSERT_EQ(last_vector.activations[0], 33.0);
}

TEST(UserCountHistoryBlockTest, NormalizeWhenLookbackPeriodsGreaterThanOne) {
  auto count_history = CountHistoryMap::make(/* lookahead_periods= */ 1,
                                             /* lookback_periods= */ 5);

  auto block =
      UserCountHistoryBlock::make(/* user_col= */ 0, /* count_col= */ 1,
                                  /* timestamp_col= */ 2, count_history);

  std::vector<std::string> input_rows = {
      {"user,0,2022-02-02"}, {"user,1,2022-02-03"}, {"user,2,2022-02-04"},
      {"user,3,2022-02-05"}, {"user,4,2022-02-06"}, {"user,5,2022-02-07"},
  };

  auto batch = processBatch(block, input_rows);

  auto last_vector = batch[5];

  ASSERT_TRUE(last_vector.isDense());
  ASSERT_EQ(last_vector.len, 5);

  float l2norm = 0.0;
  float last_act = 0.0;
  float last_dif = 0.0;

  for (uint32_t pos = 0; pos < last_vector.len; pos++) {
    float act = last_vector.activations[pos];
    l2norm += act * act;

    float dif = act - last_act;
    if (pos > 1) {
      ASSERT_NEAR(last_dif, dif, 0.00001);
    }
    last_dif = dif;
    last_act = act;
  }

  l2norm = std::sqrt(l2norm);
  ASSERT_NEAR(l2norm, 1.0, 0.00001);
}

}  // namespace thirdai::dataset
