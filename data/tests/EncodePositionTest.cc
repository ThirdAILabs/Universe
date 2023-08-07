#include <gtest/gtest.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/EncodePosition.h>
#include <data/src/transformations/State.h>
#include <vector>

namespace thirdai::data::tests {

static std::vector<std::vector<uint32_t>> encode_positions(
    std::vector<std::vector<uint32_t>> sequences) {
  auto sequence_column =
      ArrayColumn<uint32_t>::make(std::move(sequences), /* dim= */ 100000);

  ColumnMap column_map({{"sequence", sequence_column}});

  EncodePositionTransform transform(/* input_column= */ "sequence",
                                    /* output_column= */ "pos_sequence",
                                    /* hash_range= */ 100000);

  State state(/* mach_index= */ nullptr);
  column_map = transform.apply(column_map, state);

  auto pos_seq_column = column_map.getArrayColumn<uint32_t>("pos_sequence");

  std::vector<std::vector<uint32_t>> output(pos_seq_column->numRows());
  for (uint32_t i = 0; i < pos_seq_column->numRows(); i++) {
    auto row = pos_seq_column->row(i);
    output[i].insert(output[i].begin(), row.begin(), row.end());
  }

  return output;
}

TEST(EncodePositionTest, SameSequence) {
  auto position_encoded_sequences = encode_positions(/* sequences= */ {
      {0, 1, 2, 3, 4},
      {0, 1, 2, 3, 4},
  });

  for (uint32_t pos = 0; pos < 5; pos++) {
    ASSERT_EQ(position_encoded_sequences[0][pos],
              position_encoded_sequences[1][pos]);

    if (pos < 4) {
      ASSERT_NE(position_encoded_sequences[0][pos],
                position_encoded_sequences[0][pos + 1]);
    }
  }
}

/**
 * Ensures that encodings of the same tokens in the same positions are the same
 * even if the other tokens are different.
 */
TEST(EncodePositionTest, SameTokenSamePosition) {
  auto position_encoded_sequences = encode_positions(/* sequences= */ {
      {5, 1, 6, 3, 7},
      {0, 1, 2, 3, 4},
  });
  ASSERT_EQ(position_encoded_sequences[0][1], position_encoded_sequences[1][1]);
  ASSERT_EQ(position_encoded_sequences[0][3], position_encoded_sequences[1][3]);
}

TEST(EncodePositionTest, SameTokenDifferentPosition) {
  auto position_encoded_sequences = encode_positions(/* sequences= */ {
      {1, 1, 1, 1, 1},
  });

  for (uint32_t pos = 0; pos < 4; ++pos) {
    ASSERT_NE(position_encoded_sequences[0][pos],
              position_encoded_sequences[0][pos + 1]);
  }
}

TEST(EncodePositionTest, DifferentTokenSamePosition) {
  auto position_encoded_sequences = encode_positions(/* sequences= */ {
      {1, 2},
      {2, 1},
  });
  ASSERT_NE(position_encoded_sequences[0][0], position_encoded_sequences[1][0]);
  ASSERT_NE(position_encoded_sequences[0][1], position_encoded_sequences[1][1]);
}

}  // namespace thirdai::data::tests