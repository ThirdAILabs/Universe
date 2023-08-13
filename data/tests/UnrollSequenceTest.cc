#include <gtest/gtest.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/UnrollSequence.h>
#include <optional>
#include <stdexcept>
#include <vector>

namespace thirdai::data::tests {

static void assertRowsEqual(
    const ArrayColumnBase<uint32_t>& column,
    const std::vector<std::vector<uint32_t>>& expected) {
  ASSERT_EQ(column.numRows(), expected.size());
  for (uint32_t i = 0; i < column.numRows(); i++) {
    uint32_t pos = 0;
    for (uint32_t token : column.row(i)) {
      ASSERT_EQ(token, expected[i][pos]);
      pos++;
    }
  }
}

TEST(UnrollSequenceTest, DifferentRowSizesThrowsError) {
  auto source_column = ArrayColumn<uint32_t>::make(/* data= */ {{0, 1}},
                                                   /* dim= */ std::nullopt);
  auto target_column = ArrayColumn<uint32_t>::make(/* data= */ {{0, 1, 2}},
                                                   /* dim= */ std::nullopt);

  ColumnMap columns(
      /* columns= */ {{"source", source_column}, {"target", target_column}});

  UnrollSequence unroll_seq(/* source_input_column= */ "source",
                            /* target_input_column= */ "target",
                            /* source_output_column= */ "source_unrolled",
                            /* target_output_column= */ "target_unrolled");

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      unroll_seq.applyStateless(columns), std::invalid_argument);
}

TEST(UnrollSequenceTest, CorrectUnrollingSameSourceTargetColumn) {
  auto tokens =
      ArrayColumn<uint32_t>::make(/* data= */ {{0}, {1, 2}, {3, 4, 5}},
                                  /* dim= */ 100);

  ColumnMap columns(/* columns= */ {{"tokens", tokens}});

  UnrollSequence unroll_seq(/* source_input_column= */ "tokens",
                            /* target_input_column= */ "tokens",
                            /* source_output_column= */ "source_unrolled",
                            /* target_output_column= */ "target_unrolled");

  columns = unroll_seq.applyStateless(columns);

  auto source_unrolled = columns.getArrayColumn<uint32_t>("source_unrolled");
  auto target_unrolled = columns.getArrayColumn<uint32_t>("target_unrolled");

  assertRowsEqual(
      /* column= */ *source_unrolled,
      /* expected= */ {
          {},      // First unrolling of {0} (First is always empty)
          {},      // First unrolling of {1, 2} (First is always empty)
          {1},     // Second unrolling of {1, 2}
          {},      // First unrolling of {3, 4, 5} (First is always empty)
          {3},     // Second unrolling of {3, 4, 5}
          {3, 4},  // Third unrolling of {3, 4, 5}
      });

  assertRowsEqual(
      /* column= */ *target_unrolled,
      /* expected= */ {{0}, {1}, {2}, {3}, {4}, {5}});

  ASSERT_EQ(source_unrolled->dim().value(), tokens->dim().value());
  ASSERT_EQ(target_unrolled->dim().value(), tokens->dim().value());
}

TEST(UnrollSequenceTest, CorrectUnrollingDifferentSourceTargetColumn) {
  auto source =
      ArrayColumn<uint32_t>::make(/* data= */ {{0}, {1, 2}, {3, 4, 5}},
                                  /* dim= */ 100);
  auto target =
      ArrayColumn<uint32_t>::make(/* data= */ {{6}, {7, 8}, {9, 10, 11}},
                                  /* dim= */ 100);

  ColumnMap columns(/* columns= */ {{"source", source}, {"target", target}});

  UnrollSequence unroll_seq(/* source_input_column= */ "source",
                            /* target_input_column= */ "target",
                            /* source_output_column= */ "source_unrolled",
                            /* target_output_column= */ "target_unrolled");

  columns = unroll_seq.applyStateless(columns);

  auto source_unrolled = columns.getArrayColumn<uint32_t>("source_unrolled");
  auto target_unrolled = columns.getArrayColumn<uint32_t>("target_unrolled");

  assertRowsEqual(
      /* column= */ *source_unrolled,
      /* expected= */ {
          {},      // First unrolling of {0} (First is always empty)
          {},      // First unrolling of {1, 2} (First is always empty)
          {1},     // Second unrolling of {1, 2}
          {},      // First unrolling of {3, 4, 5} (First is always empty)
          {3},     // Second unrolling of {3, 4, 5}
          {3, 4},  // Third unrolling of {3, 4, 5}
      });

  assertRowsEqual(
      /* column= */ *target_unrolled,
      /* expected= */ {{6}, {7}, {8}, {9}, {10}, {11}});

  ASSERT_EQ(source_unrolled->dim().value(), source->dim().value());
  ASSERT_EQ(target_unrolled->dim().value(), target->dim().value());
}

}  // namespace thirdai::data::tests