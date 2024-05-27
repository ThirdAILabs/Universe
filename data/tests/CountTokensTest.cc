#include <gtest/gtest.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/CountTokens.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace thirdai::data::tests {

static void assertRowsEqual(RowView<uint32_t> row,
                            const std::vector<uint32_t>& expected) {
  std::vector<uint32_t> row_vec(row.begin(), row.end());
  ASSERT_EQ(row_vec, expected);
}

static void assertCorrectCounts(const Transformation& transform,
                                std::vector<std::vector<uint32_t>> tokens_data,
                                std::optional<uint32_t> max_tokens) {
  auto tokens_data_copy = tokens_data;
  auto tokens_column =
      ArrayColumn<uint32_t>::make(/* data= */ std::move(tokens_data),
                                  /* dim= */ std::nullopt);

  ColumnMap columns({{"tokens", tokens_column}});

  columns = transform.applyStateless(columns);

  auto final_tokens_column = columns.getArrayColumn<uint32_t>("tokens");
  ASSERT_EQ(tokens_column, final_tokens_column);
  auto count_column = columns.getValueColumn<uint32_t>("count");

  for (uint32_t i = 0; i < columns.numRows(); i++) {
    // Make sure we didn't change the tokens column
    assertRowsEqual(final_tokens_column->row(i), tokens_data_copy.at(i));
    // Check counts are correct.
    uint32_t num_tokens = tokens_data_copy.at(i).size();
    uint32_t expected_count =
        max_tokens ? std::min(*max_tokens, num_tokens) : num_tokens;
    ASSERT_EQ(count_column->value(i), expected_count);
  }

  if (max_tokens) {
    ASSERT_EQ(count_column->dim().value(), max_tokens.value() + 1);
  }
}

TEST(CountTokensTest, CorrectCountsNoMaxTokens) {
  CountTokens counter(/* input_column= */ "tokens",
                      /* output_column= */ "count",
                      /* max_tokens= */ std::nullopt);

  assertCorrectCounts(
      /* transform= */ counter, /* tokens_data= */
      {{}, {0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 3}, {0, 1, 2, 3, 4}},
      /* max_tokens= */ std::nullopt);
}

TEST(CountTokensTest, CorrectCountsWithMaxTokens) {
  CountTokens counter(/* input_column= */ "tokens",
                      /* output_column= */ "count",
                      /* max_tokens= */ 3);

  assertCorrectCounts(
      /* transform= */ counter, /* tokens_data= */
      {{}, {0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 3}, {0, 1, 2, 3, 4}},
      /* max_tokens= */ 3);
}

TEST(CountTokensTest, CorrectCountsNoMaxTokensSerialization) {
  CountTokens counter(/* input_column= */ "tokens",
                      /* output_column= */ "count",
                      /* max_tokens= */ std::nullopt);

  auto new_counter = Transformation::deserialize(counter.serialize());

  assertCorrectCounts(
      /* transform= */ *new_counter, /* tokens_data= */
      {{}, {0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 3}, {0, 1, 2, 3, 4}},
      /* max_tokens= */ std::nullopt);
}

TEST(CountTokensTest, CorrectCountsWithMaxTokensSerialization) {
  CountTokens counter(/* input_column= */ "tokens",
                      /* output_column= */ "count",
                      /* max_tokens= */ 3);

  auto new_counter = Transformation::deserialize(counter.serialize());

  assertCorrectCounts(
      /* transform= */ *new_counter, /* tokens_data= */
      {{}, {0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 3}, {0, 1, 2, 3, 4}},
      /* max_tokens= */ 3);
}

}  // namespace thirdai::data::tests