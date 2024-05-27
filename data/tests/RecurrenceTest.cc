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
      /* max_sequence_length= */ 100);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      recurrence.applyStateless(columns), std::invalid_argument);
}

TEST(RecurrenceTest, CorrectUnrollingSameSourceTargetColumn) {
  uint32_t TOKEN_VOCAB_SIZE = 99;
  uint32_t MAX_SEQ_LEN = 10;

  auto tokens =
      ArrayColumn<uint32_t>::make(/* data= */ {{0}, {1, 2}, {3, 4, 5}},
                                  /* dim= */ TOKEN_VOCAB_SIZE);

  ColumnMap columns(/* columns= */ {{"tokens", tokens}});

  Recurrence recurrence(
      /* source_input_column= */ "tokens",
      /* target_input_column= */ "tokens",
      /* source_output_column= */ "source_unrolled",
      /* target_output_column= */ "target_unrolled",
      /* target_vocab_size= */ TOKEN_VOCAB_SIZE,
      /* max_sequence_length= */ MAX_SEQ_LEN);

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
      // vocab_size = 99 is EOS token before position encoding
      /* expected= */ {
          {0},    // pos * (vocab_size + 1) + token = 0 * 100 + 0 = 1
          {199},  // EOS. pos * (vocab_size + 1) + token = 1 * 100 + 99 = 199
          {1},    // pos * (vocab_size + 1) + token = 0 * 100 + 1 = 1
          {102},  // pos * (vocab_size + 1) + token = 1 * 100 + 2 = 102
          {299},  // EOS. pos * (vocab_size + 1) + token = 2 * 100 + 99 = 299
          {3},    // pos * (vocab_size + 1) + token = 0 * 100 + 3 = 3
          {104},  // pos * (vocab_size + 1) + token = 1 * 100 + 4 = 104
          {205},  // pos * (vocab_size + 1) + token = 2 * 100 + 5 = 205
          {399},  // pos * (vocab_size + 1) + token = 3 * 100 + 99 = 399
      });

  for (uint32_t i = 0; i < (TOKEN_VOCAB_SIZE + 1) * MAX_SEQ_LEN; i++) {
    if (i % (TOKEN_VOCAB_SIZE + 1) == TOKEN_VOCAB_SIZE) {
      ASSERT_TRUE(recurrence.isEOS(i));
    } else {
      ASSERT_FALSE(recurrence.isEOS(i));
    }
  }

  ASSERT_EQ(source_unrolled->dim(), tokens->dim());
  ASSERT_EQ(target_unrolled->dim(), (TOKEN_VOCAB_SIZE + 1) * MAX_SEQ_LEN);
}

TEST(RecurrenceTest, CorrectUnrollingDifferentSourceTargetColumn) {
  uint32_t TOKEN_VOCAB_SIZE = 99;
  uint32_t MAX_SEQ_LEN = 10;

  auto source =
      ArrayColumn<uint32_t>::make(/* data= */ {{0}, {1, 2}, {3, 4, 5}},
                                  /* dim= */ std::nullopt);
  auto target =
      ArrayColumn<uint32_t>::make(/* data= */ {{6}, {7, 8}, {9, 10, 11}},
                                  /* dim= */ TOKEN_VOCAB_SIZE);

  ColumnMap columns(/* columns= */ {{"source", source}, {"target", target}});

  Recurrence recurrence(
      /* source_input_column= */ "source",
      /* target_input_column= */ "target",
      /* source_output_column= */ "source_unrolled",
      /* target_output_column= */ "target_unrolled",
      /* target_vocab_size= */ TOKEN_VOCAB_SIZE,
      /* max_sequence_length= */ MAX_SEQ_LEN);

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
      // vocab_size = 99 is EOS token before position encoding
      /* expected= */ {
          {6},    // pos * (vocab_size + 1) + token = 0 * 100 + 6 = 6
          {199},  // EOS. pos * (vocab_size + 1) + token = 1 * 100 + 99 = 199
          {7},    // pos * (vocab_size + 1) + token = 0 * 100 + 7 = 7
          {108},  // pos * (vocab_size + 1) + token = 1 * 100 + 8 = 108
          {299},  // EOS. pos * (vocab_size + 1) + token = 2 * 100 + 99 = 299
          {9},    // pos * (vocab_size + 1) + token = 0 * 100 + 9 = 9
          {110},  // pos * (vocab_size + 1) + token = 1 * 100 + 10 = 110
          {211},  // pos * (vocab_size + 1) + token = 2 * 100 + 11 = 211
          {399},  // pos * (vocab_size + 1) + token = 3 * 100 + 99 = 399
      });

  for (uint32_t i = 0; i < (TOKEN_VOCAB_SIZE + 1) * MAX_SEQ_LEN; i++) {
    if (i % (TOKEN_VOCAB_SIZE + 1) == TOKEN_VOCAB_SIZE) {
      ASSERT_TRUE(recurrence.isEOS(i));
    } else {
      ASSERT_FALSE(recurrence.isEOS(i));
    }
  }

  ASSERT_EQ(source_unrolled->dim(), source->dim());
  ASSERT_EQ(target_unrolled->dim(), (TOKEN_VOCAB_SIZE + 1) * MAX_SEQ_LEN);
}

TEST(RecurrenceTest, CorrectUnrollingWithSequencesLongerThanMaxSequenceLength) {
  uint32_t TOKEN_VOCAB_SIZE = 99;
  uint32_t MAX_SEQ_LEN = 2;

  auto source =
      ArrayColumn<uint32_t>::make(/* data= */ {{}, {0}, {1, 2}, {3, 4, 5}},
                                  /* dim= */ std::nullopt);
  auto target =
      ArrayColumn<uint32_t>::make(/* data= */ {{}, {6}, {7, 8}, {9, 10, 11}},
                                  /* dim= */ TOKEN_VOCAB_SIZE);

  ColumnMap columns(/* columns= */ {{"source", source}, {"target", target}});

  Recurrence recurrence(
      /* source_input_column= */ "source",
      /* target_input_column= */ "target",
      /* source_output_column= */ "source_unrolled",
      /* target_output_column= */ "target_unrolled",
      /* target_vocab_size= */ TOKEN_VOCAB_SIZE,
      /* max_sequence_length= */ MAX_SEQ_LEN);

  columns = recurrence.applyStateless(columns);

  auto source_unrolled = columns.getArrayColumn<uint32_t>("source_unrolled");
  auto target_unrolled = columns.getArrayColumn<uint32_t>("target_unrolled");

  assertRowsEqual(
      /* column= */ *source_unrolled,
      /* expected= */ {
          {},   // First unrolling of {}
          {},   // First unrolling of {0} (First is always empty)
          {0},  // Last unrolling of {0} (Whole sequence, next token is EOS)
          {},   // First unrolling of {1, 2} (First is always empty)
          {1},  // Second unrolling of {1, 2}
          // Skip last unrolling of {1, 2} since we already predicted 2 tokens.
          {},   // First unrolling of {3, 4, 5}
          {3},  // Second unrolling of {3, 4, 5}
                // Skip third and last unrolling of {3, 4, 5} since we already
                // predicted 2 tokens.
      });

  assertRowsEqual(
      /* column= */ *target_unrolled,
      /* expected= */ {
          // From sequence {}
          {99},  // EOS. pos * (vocab_size + 1) + token = 0 * 100 + 99 = 99
          // From sequence {6}
          {6},    // pos * (vocab_size + 1) + token = 0 * 100 + 6 = 6
          {199},  // EOS. pos * (vocab_size + 1) + token = 1 * 100 + 99 = 199
          // From sequence {7, 8}
          {7},    // pos * (vocab_size + 1) + token = 0 * 100 + 7 = 7
          {108},  // pos * (vocab_size + 1) + token = 1 * 100 + 8 = 108
          // Skip EOS at third position (pos=2) since max seq len is 2
          // From sequence {9, 10, 11}
          {9},    // pos * (vocab_size + 1) + token = 0 * 100 + 9 = 9
          {110},  // pos * (vocab_size + 1) + token = 1 * 100 + 10 = 110
                  // Skip third element and EOS since max seq len is 2
      });

  for (uint32_t i = 0; i < (TOKEN_VOCAB_SIZE + 1) * MAX_SEQ_LEN; i++) {
    if (i % (TOKEN_VOCAB_SIZE + 1) == TOKEN_VOCAB_SIZE) {
      ASSERT_TRUE(recurrence.isEOS(i));
    } else {
      ASSERT_FALSE(recurrence.isEOS(i));
    }
  }

  ASSERT_EQ(source_unrolled->dim(), source->dim());
  ASSERT_EQ(target_unrolled->dim(), (TOKEN_VOCAB_SIZE + 1) * MAX_SEQ_LEN);
}

TEST(RecurrenceTest, Serialization) {
  uint32_t TOKEN_VOCAB_SIZE = 99;
  uint32_t MAX_SEQ_LEN = 3;

  auto source =
      ArrayColumn<uint32_t>::make(/* data= */ {{1, 2}, {3, 4, 5}, {6}},
                                  /* dim= */ std::nullopt);
  auto target =
      ArrayColumn<uint32_t>::make(/* data= */ {{10, 20}, {30, 40, 50}, {60}},
                                  /* dim= */ TOKEN_VOCAB_SIZE);

  ColumnMap columns(/* columns= */ {{"source", source}, {"target", target}});

  Recurrence recurrence(
      /* source_input_column= */ "source",
      /* target_input_column= */ "target",
      /* source_output_column= */ "source_unrolled",
      /* target_output_column= */ "target_unrolled",
      /* target_vocab_size= */ TOKEN_VOCAB_SIZE,
      /* max_sequence_length= */ MAX_SEQ_LEN);

  // We down cast to transformation because otherwise it was trying to call
  // the cereal "serialize" method. This can be removed once cereal is
  // officially depreciated.
  auto new_recurrence = Transformation::deserialize(
      dynamic_cast<Transformation*>(&recurrence)->serialize());

  auto output = new_recurrence->applyStateless(columns);

  assertRowsEqual(*output.getArrayColumn<uint32_t>("source_unrolled"),
                  {{}, {1}, {1, 2}, {}, {3}, {3, 4}, {}, {6}});

  assertRowsEqual(*output.getArrayColumn<uint32_t>("target_unrolled"),
                  {{10}, {120}, {299}, {30}, {140}, {250}, {60}, {199}});
}

}  // namespace thirdai::data::tests