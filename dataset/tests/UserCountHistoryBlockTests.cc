#include <gtest/gtest.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/UserCountHistory.h>
#include <dataset/src/featurizers/GenericFeaturizer.h>
#include <dataset/src/featurizers/ProcessorUtils.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <dataset/src/utils/TimeUtils.h>
#include <cmath>
#include <string_view>

namespace thirdai::dataset {

static BoltBatch processBatch(BlockPtr block,
                              const std::vector<std::string>& input_rows) {
  GenericFeaturizer processor(
      /* input_blocks= */ {std::move(block)}, /* label_blocks= */ {},
      /* has_header= */ false, /* delimiter= */ ',', /* parallel= */ false);
  auto batch = processor.createBatch(input_rows).at(0);
  return std::move(batch);
}

TEST(UserCountHistoryBlockTest, ExplanationWorks) {
  auto count_history = QuantityHistoryTracker::make(/* history_lag= */ 1,
                                                    /* history_length= */ 5);

  auto block = UserCountHistoryBlock::make(
      /* user_col= */ 0, /* count_col= */ 1,
      /* timestamp_col= */ 2, count_history,
      /* should_update_history= */ true, /* include_current_row= */ true);

  std::vector<std::string> input_rows = {
      {"user,0,2022-02-02"}, {"user,1,2022-02-03"}, {"user,2,2022-02-04"},
      {"user,3,2022-02-05"}, {"user,4,2022-02-06"}, {"user,5,2022-02-07"},
  };

  auto batch = processBatch(block, input_rows);

  CsvSampleRef input_row(input_rows[5], ',');

  auto explanation_0 =
      block->explainIndex(/* index_within_block= */ 0, input_row);
  ASSERT_EQ(explanation_0.column_number,
            1);  // Count is in column 2 of input row
  ASSERT_EQ(explanation_0.keyword,
            "between 2022-02-02 and 2022-02-03 value is lower than usual");

  auto explanation_2 =
      block->explainIndex(/* index_within_block= */ 2, input_row);
  ASSERT_EQ(explanation_2.column_number,
            1);  // Count is in column 2 of input row
  ASSERT_EQ(explanation_2.keyword,
            "between 2022-02-04 and 2022-02-05 value is same as usual");

  auto explanation_4 =
      block->explainIndex(/* index_within_block= */ 4, input_row);
  ASSERT_EQ(explanation_4.column_number,
            1);  // Count is in column 2 of input row
  ASSERT_EQ(explanation_4.keyword,
            "between 2022-02-06 and 2022-02-07 value is higher than usual");
}

TEST(UserCountHistoryBlockTest, NoNormalizeWhenLookbackPeriodsEqualsOne) {
  auto count_history = QuantityHistoryTracker::make(/* history_lag= */ 1,
                                                    /* history_length= */ 1);

  auto block = UserCountHistoryBlock::make(
      /* user_col= */ 0, /* count_col= */ 1,
      /* timestamp_col= */ 2, count_history,
      /* should_update_history= */ true, /* include_current_row= */ true);

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
  auto count_history = QuantityHistoryTracker::make(/* history_lag= */ 1,
                                                    /* history_length= */ 5);

  auto block = UserCountHistoryBlock::make(
      /* user_col= */ 0, /* count_col= */ 1,
      /* timestamp_col= */ 2, count_history,
      /* should_update_history= */ true, /* include_current_row= */ true);

  std::vector<std::string> input_rows = {
      {"user,0,2022-02-02"}, {"user,1,2022-02-03"}, {"user,2,2022-02-04"},
      {"user,3,2022-02-05"}, {"user,4,2022-02-06"}, {"user,5,2022-02-07"},
  };

  auto batch = processBatch(block, input_rows);

  // First vector's raw counts should be 0. Ensure that it is not normalized to
  // nan.
  auto first_vector = batch[0];

  ASSERT_TRUE(first_vector.isDense());
  ASSERT_EQ(first_vector.len, 5);
  for (uint32_t pos = 0; pos < first_vector.len; pos++) {
    ASSERT_FALSE(std::isnan(first_vector.activations[pos]));
  }

  // Last vector should have counts from samples 0-4.
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

TEST(UserCountHistoryBlockTest, NotNumbersTreatedAsZero) {
  auto count_history = QuantityHistoryTracker::make(/* history_lag= */ 0,
                                                    /* history_length= */ 1);

  auto block = UserCountHistoryBlock::make(
      /* user_col= */ 0, /* count_col= */ 1,
      /* timestamp_col= */ 2, count_history,
      /* should_update_history= */ true, /* include_current_row= */ true);

  std::vector<std::string> input_rows = {
      {"user,nan,2022-02-02"},
      {"user,NaN,2022-02-03"},
      {"user,inf,2022-02-04"},
      {"user,-inf,2022-02-05"},
  };

  auto batch = processBatch(block, input_rows);

  for (const auto& vector : batch) {
    ASSERT_EQ(vector.len, 1);
    ASSERT_EQ(vector.activations[0], 0.0);
  }
}

TEST(UserCountHistoryBlockTest,
     HistoryDoesNotChangeWhenShouldUpdateHistoryIsFalse) {
  auto count_history = QuantityHistoryTracker::make(/* history_lag= */ 0,
                                                    /* history_length= */ 1);

  auto block = UserCountHistoryBlock::make(
      /* user_col= */ 0, /* count_col= */ 1,
      /* timestamp_col= */ 2, count_history,
      /* should_update_history= */ false, /* include_current_row= */ true);

  std::string key = "user";
  std::string val = "5";
  std::string timestamp = "2022-02-02";

  std::vector<std::string> input_rows = {
      {key + "," + val + "," + timestamp},
  };

  auto count_before = count_history->getHistory(
      key, TimeObject(std::string_view(timestamp.data())).secondsSinceEpoch());

  processBatch(block, input_rows);

  std::vector<std::string_view> input_row_view(3);
  input_row_view[0] = std::string_view(key.data(), /* len= */ 4);
  input_row_view[1] = std::string_view(val.data(), /* len= */ 1);
  input_row_view[2] = std::string_view(timestamp.data(), /* len= */ 10);

  RowSampleRef input_row_view_ref(input_row_view);
  block->explainIndex(/* index_within_block= */ 0, input_row_view_ref);

  auto count_after =
      count_history->getHistory(key, TimeObject(timestamp).secondsSinceEpoch());

  ASSERT_EQ(count_before[0], count_after[0]);
}

TEST(UserCountHistoryBlockTest, IncludeCurrentRowFlagWorks) {
  // Make sure that if include_current_row is false, a row's count column
  // is not included in the corresponding vector.
  auto count_history = QuantityHistoryTracker::make(/* history_lag= */ 0,
                                                    /* history_length= */ 3);

  auto block = UserCountHistoryBlock::make(
      /* user_col= */ 0, /* count_col= */ 1,
      /* timestamp_col= */ 2, count_history,
      /* should_update_history= */ true, /* include_current_row= */ false);

  std::vector<std::string> input_rows = {
      {"user,1,2022-02-02"},
      {"user,5,2022-02-04"},
      {"user,4,2022-02-05"},
  };

  auto batch = processBatch(block, input_rows);

  auto last_vector = batch[2];

  // First make sure that the the exclude current row block only excludes
  // the current row but still tracks previous over time.
  for (uint32_t i = 0; i < 3; i++) {
    // We can be sure that none of the counts are zero because
    // the average of 1, 5, and 0 is 2 (not 1, 5, or 0)
    ASSERT_NE(last_vector.activations[i], 0);
  }

  /*
    Since we exclude the current row, the count on 2022-02-05 is 0, which
    is lower than average. Therefore, we expect that the activation at
    position 2 is < 0 after normalization.
  */
  ASSERT_LT(last_vector.activations[2], 0);
}

// TODO(Geordie): Test that if include_current_row == true, then vector includes
// last item, and not otherwise, regardless of should_update_history

}  // namespace thirdai::dataset
