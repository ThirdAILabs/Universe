#include <gtest/gtest.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/CountTokens.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace thirdai::data::tests {

static void assertRowsEqual(RowView<uint32_t> row_1, RowView<uint32_t> row_2) {
  ASSERT_EQ(row_1.size(), row_2.size());
  std::vector<uint32_t> row_1_vec(row_1.begin(), row_1.end());
  std::vector<uint32_t> row_2_vec(row_2.begin(), row_2.end());
  for (uint32_t i = 0; i < row_1.size(); ++i) {
    ASSERT_EQ(row_1_vec[i], row_2_vec[i]);
  }
}

static void assertCorrectCounts(std::vector<std::vector<uint32_t>> tokens_data,
                                std::optional<uint32_t> ceiling) {
  auto tokens_data_copy = tokens_data;
  auto tokens_column =
      ArrayColumn<uint32_t>::make(/* data= */ std::move(tokens_data),
                                  /* dim= */ std::nullopt);

  // We create an independent copy to help us assert that the tokens column is
  // not changed by the transformation
  auto tokens_column_copy =
      ArrayColumn<uint32_t>::make(/* data= */ std::move(tokens_data_copy),
                                  /* dim= */ std::nullopt);

  ColumnMap columns({{"tokens", tokens_column}});

  CountTokens counter(/* input_column= */ "tokens",
                      /* output_column= */ "count",
                      /* ceiling= */ ceiling);

  columns = counter.applyStateless(columns);

  auto final_tokens_column = columns.getArrayColumn<uint32_t>("tokens");
  auto count_column = columns.getValueColumn<uint32_t>("count");

  for (uint32_t i = 0; i < columns.numRows(); ++i) {
    // Make sure we didn't change the tokens column
    assertRowsEqual(tokens_column_copy->row(i), final_tokens_column->row(i));
    // Check counts are correct.
    uint32_t num_tokens = tokens_column->row(i).size();
    uint32_t expected_count =
        ceiling ? std::min(*ceiling, num_tokens) : num_tokens;
    ASSERT_EQ(count_column->value(i), expected_count);
  }
}

TEST(CountTokensTest, CorrectCountsNoCeiling) {
  assertCorrectCounts(/* tokens_data= */
                      {{},
                       {0},
                       {0, 1},
                       {0, 1, 2},
                       {0, 1, 2, 3},
                       {0, 1, 2, 3, 4}},
                      /* ceiling= */ std::nullopt);
}

TEST(CountTokensTest, CorrectCountsWithCeiling) {
  assertCorrectCounts(/* tokens_data= */
                      {{},
                       {0},
                       {0, 1},
                       {0, 1, 2},
                       {0, 1, 2, 3},
                       {0, 1, 2, 3, 4}},
                      /* ceiling= */ 3);
}

}  // namespace thirdai::data::tests