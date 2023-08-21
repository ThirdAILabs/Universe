#include <gtest/gtest.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/Recurrence.h>
#include <cstddef>
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

TEST(RecurrenceTest, DifferentRowSizesThrowsError) {
  size_t vocab_size = 10;

  auto source_column = ArrayColumn<uint32_t>::make(/* data= */ {{0, 1}},
                                                   /* dim= */ std::nullopt);
  auto target_column = ArrayColumn<uint32_t>::make(/* data= */ {{0, 1, 2}},
                                                   /* dim= */ vocab_size);

  ColumnMap columns(
      /* columns= */ {{"source", source_column}, {"target", target_column}});

  Recurrence recurrence(
      /* source_input_column= */ "source",
      /* target_input_column= */ "target",
      /* source_output_column= */ "source_unrolled",
      /* target_output_column= */ "target_unrolled",
      /* target_vocab_size= */ vocab_size,
      /* max_positions= */ 1);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      recurrence.applyStateless(columns), std::invalid_argument);
}

TEST(RecurrenceTest, CorrectUnrollingSameSourceTargetColumn) {
  auto tokens =
      ArrayColumn<uint32_t>::make(/* data= */ {{0}, {1, 2}, {3, 4, 5}},
                                  /* dim= */ 100);

  ColumnMap columns(/* columns= */ {{"tokens", tokens}});

  Recurrence recurrence(
      /* source_input_column= */ "tokens",
      /* target_input_column= */ "tokens",
      /* source_output_column= */ "source_unrolled",
      /* target_output_column= */ "target_unrolled",
      /* target_vocab_size= */ 100,
      /* max_positions= */ 1);

  columns = recurrence.applyStateless(columns);

  auto source_unrolled = columns.getArrayColumn<uint32_t>("source_unrolled");
  auto target_unrolled = columns.getArrayColumn<uint32_t>("target_unrolled");

  assertRowsEqual(
      /* column= */ *source_unrolled,
      /* expected= */ {
          {},      // First unrolling of {0} (First is always empty)
          {0},     // Last unrolling of {0} (Whole sequence, next token is EOS)
          {},      // First unrolling of {1, 2} (First is always empty)
          {1},     // Second unrolling of {1, 2}
          {1, 2},  // Last unrolling of {1, 2} (Whole sequence, next is EOS)
          {},      // First unrolling of {3, 4, 5}
          {3},     // Second unrolling of {3, 4, 5}
          {3, 4},  // Third unrolling of {3, 4, 5}
          {3, 4, 5},  // Last unrolling of {3, 4, 5}
      });

  assertRowsEqual(
      /* column= */ *target_unrolled,
      // 100 is EOS token.
      /* expected= */ {{0}, {100}, {1}, {2}, {100}, {3}, {4}, {5}, {100}});

  ASSERT_TRUE(recurrence.isEOS(100));
  for (uint32_t i = 0; i < 100; i++) {
    ASSERT_FALSE(recurrence.isEOS(i));
  }

  ASSERT_EQ(source_unrolled->dim(), tokens->dim());
  ASSERT_EQ(target_unrolled->dim(), *tokens->dim() + 1);
}

TEST(RecurrenceTest, CorrectUnrollingDifferentSourceTargetColumn) {
  auto source =
      ArrayColumn<uint32_t>::make(/* data= */ {{0}, {1, 2}, {3, 4, 5}},
                                  /* dim= */ std::nullopt);
  auto target =
      ArrayColumn<uint32_t>::make(/* data= */ {{6}, {7, 8}, {9, 10, 11}},
                                  /* dim= */ 100);

  ColumnMap columns(/* columns= */ {{"source", source}, {"target", target}});

  Recurrence recurrence(
      /* source_input_column= */ "source",
      /* target_input_column= */ "target",
      /* source_output_column= */ "source_unrolled",
      /* target_output_column= */ "target_unrolled",
      /* target_vocab_size= */ 100,
      /* max_positions= */ 1);

  columns = recurrence.applyStateless(columns);

  auto source_unrolled = columns.getArrayColumn<uint32_t>("source_unrolled");
  auto target_unrolled = columns.getArrayColumn<uint32_t>("target_unrolled");

  assertRowsEqual(
      /* column= */ *source_unrolled,
      /* expected= */ {
          {},      // First unrolling of {0} (First is always empty)
          {0},     // Last unrolling of {0} (Whole sequence, next token is EOS)
          {},      // First unrolling of {1, 2} (First is always empty)
          {1},     // Second unrolling of {1, 2}
          {1, 2},  // Last unrolling of {1, 2} (Whole sequence, next is EOS)
          {},      // First unrolling of {3, 4, 5}
          {3},     // Second unrolling of {3, 4, 5}
          {3, 4},  // Third unrolling of {3, 4, 5}
          {3, 4, 5},  // Last unrolling of {3, 4, 5}
      });

  assertRowsEqual(
      /* column= */ *target_unrolled,
      // 100 is EOS token.
      /* expected= */ {{6}, {100}, {7}, {8}, {100}, {9}, {10}, {11}, {100}});

  ASSERT_TRUE(recurrence.isEOS(100));
  for (uint32_t i = 0; i < 100; i++) {
    ASSERT_FALSE(recurrence.isEOS(i));
  }

  ASSERT_EQ(source_unrolled->dim(), source->dim());
  ASSERT_EQ(target_unrolled->dim(), *target->dim() + 1);
}

TEST(RecurrenceTest,
     CorrectUnrollingDifferentSourceTargetColumnWithPositionalOffsets) {
  auto source =
      ArrayColumn<uint32_t>::make(/* data= */ {{}, {0}, {1, 2}, {3, 4, 5}},
                                  /* dim= */ 100);
  auto target =
      ArrayColumn<uint32_t>::make(/* data= */ {{}, {6}, {7, 8}, {9, 10, 11}},
                                  /* dim= */ 100);

  ColumnMap columns(/* columns= */ {{"source", source}, {"target", target}});

  Recurrence recurrence(
      /* source_input_column= */ "source",
      /* target_input_column= */ "target",
      /* source_output_column= */ "source_unrolled",
      /* target_output_column= */ "target_unrolled",
      /* target_vocab_size= */ 100,
      /* max_positions= */ 2);

  columns = recurrence.applyStateless(columns);

  auto source_unrolled = columns.getArrayColumn<uint32_t>("source_unrolled");
  auto target_unrolled = columns.getArrayColumn<uint32_t>("target_unrolled");

  assertRowsEqual(
      /* column= */ *source_unrolled,
      /* expected= */ {
          {},      // First unrolling of {}
          {},      // First unrolling of {0} (First is always empty)
          {0},     // Last unrolling of {0} (Whole sequence, next token is EOS)
          {},      // First unrolling of {1, 2} (First is always empty)
          {1},     // Second unrolling of {1, 2}
          {1, 2},  // Last unrolling of {1, 2} (Whole sequence, next is EOS)
          {},      // First unrolling of {3, 4, 5}
          {3},     // Second unrolling of {3, 4, 5}
          {3, 4},  // Third unrolling of {3, 4, 5}
          {3, 4, 5},  // Last unrolling of {3, 4, 5}
      });

  assertRowsEqual(
      /* column= */ *target_unrolled,
      /* expected= */ {
          // pos is at most 1 since max_positions = 2
          // from the sequence {}
          {200},  // EOS = max_positions * vocab_size + pos = 2 * 100 + 0
          // from the sequence {6}
          {6},    // pos * vocab_size + token = 0 * 100 + 6 = 6
          {201},  // EOS = max_positions * vocab_size + pos = 2 * 100 + 1
          // from the sequence {7, 8}
          {7},    // pos * vocab_size + token = 0 * 100 + 7 = 7
          {108},  // pos * vocab_size + token = 1 * 100 + 8 = 108
          {201},  // EOS = max_positions * vocab_size + pos = 2 * 100 + 1
          // from the sequence {9, 10, 11}
          {9},    // pos * vocab_size + token = 0 * 100 + 9 = 9
          {110},  // pos * vocab_size + token = 1 * 100 + 10 = 110
          {111},  // pos * vocab_size + token = 1 * 100 + 11 = 111
          {201},  // EOS = max_positions * vocab_size + pos = 2 * 100 + 1
      });

  ASSERT_TRUE(recurrence.isEOS(200));
  ASSERT_TRUE(recurrence.isEOS(201));
  for (uint32_t i = 0; i < 200; i++) {
    ASSERT_FALSE(recurrence.isEOS(i));
  }

  ASSERT_EQ(source_unrolled->dim(), source->dim());
  ASSERT_EQ(target_unrolled->dim(), 2 * (*target->dim() + 1));
}

}  // namespace thirdai::data::tests