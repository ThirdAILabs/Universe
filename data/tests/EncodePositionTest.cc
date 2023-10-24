#include <gtest/gtest.h>
#include <_types/_uint32_t.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/EncodePosition.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
#include <unordered_set>
#include <vector>

namespace thirdai::data::tests {

static std::vector<std::vector<uint32_t>> hashPositions(
    std::vector<std::vector<uint32_t>> sequences) {
  auto sequence_column =
      ArrayColumn<uint32_t>::make(std::move(sequences), /* dim= */ 100000);

  ColumnMap column_map({{"sequence", sequence_column}});

  HashPositionTransform transform(/* input_column= */ "sequence",
                                  /* output_column= */ "pos_sequence",
                                  /* hash_range= */ 100000);

  column_map = transform.applyStateless(column_map);

  auto pos_seq_column = column_map.getArrayColumn<uint32_t>("pos_sequence");

  std::vector<std::vector<uint32_t>> output(pos_seq_column->numRows());
  for (uint32_t i = 0; i < pos_seq_column->numRows(); i++) {
    auto row = pos_seq_column->row(i);
    output[i].insert(output[i].begin(), row.begin(), row.end());
  }

  return output;
}

TEST(EncodePositionTest, HashPositionSameSequence) {
  auto position_encoded_sequences = hashPositions(/* sequences= */ {
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
TEST(EncodePositionTest, HashPositionSameTokenSamePosition) {
  auto position_encoded_sequences = hashPositions(/* sequences= */ {
      {5, 1, 6, 3, 7},
      {0, 1, 2, 3, 4},
  });
  ASSERT_EQ(position_encoded_sequences[0][1], position_encoded_sequences[1][1]);
  ASSERT_EQ(position_encoded_sequences[0][3], position_encoded_sequences[1][3]);
}

TEST(EncodePositionTest, HashPositionSameTokenDifferentPosition) {
  auto position_encoded_sequence = hashPositions(
      /* sequences= */ {{1, 1, 1, 1, 1}})[0];

  std::unordered_set<uint32_t> unique_tokens(position_encoded_sequence.begin(),
                                             position_encoded_sequence.end());

  ASSERT_EQ(unique_tokens.size(), 5);
}

TEST(EncodePositionTest, HashPositionDifferentTokenSamePosition) {
  auto position_encoded_sequences = hashPositions(/* sequences= */ {
      {1, 2},
      {2, 1},
  });
  ASSERT_NE(position_encoded_sequences[0][0], position_encoded_sequences[1][0]);
  ASSERT_NE(position_encoded_sequences[0][1], position_encoded_sequences[1][1]);
}

TEST(EncodePositionTest, OffsetPosition) {
  auto tokens =
      ArrayColumn<uint32_t>::make(/* data= */ {{4, 3, 2, 1, 0}}, /* dim= */ 5);
  ColumnMap columns({{"tokens", tokens}});
  OffsetPositionTransform offset(/* input_column= */ "tokens",
                                 /* output_column= */ "tokens_offset",
                                 /* max_num_tokens= */ 4);
  columns = offset.applyStateless(columns);

  auto tokens_offset = columns.getArrayColumn<uint32_t>("tokens_offset");

  std::vector<uint32_t> tokens_offset_vec(tokens_offset->row(0).begin(),
                                          tokens_offset->row(0).end());

  ASSERT_EQ(tokens_offset_vec[0], 4);   // 0 * 5 + 4
  ASSERT_EQ(tokens_offset_vec[1], 8);   // 1 * 5 + 3
  ASSERT_EQ(tokens_offset_vec[2], 12);  // 2 * 5 + 2
  ASSERT_EQ(tokens_offset_vec[3], 16);  // 3 * 5 + 1
  // For the next one, position is 3 instead of 4 since max_num_tokens is 4.
  ASSERT_EQ(tokens_offset_vec[4], 15);  // 3 * 5 + 0.
}

void testEncodePositionSerialization(bool hashed_position) {
  std::vector<std::vector<uint32_t>> sequences = {
      {1, 3, 2, 0}, {3, 1}, {0, 2, 1}};

  ColumnMap columns({{"sequence", ArrayColumn<uint32_t>::make(
                                      std::move(sequences), /* dim= */ 4)}});

  TransformationPtr transform;
  if (hashed_position) {
    transform = std::make_shared<HashPositionTransform>(
        /* input_column= */ "sequence",
        /* output_column= */ "encoded",
        /* hash_range= */ 100000);
  } else {
    transform = std::make_shared<OffsetPositionTransform>(
        /* input_column= */ "sequence",
        /* output_column= */ "encoded",
        /* max_tokens= */ 3);
  }

  auto original_output = transform->applyStateless(columns);
  auto original_column = original_output.getArrayColumn<uint32_t>("encoded");

  auto new_transform = Transformation::deserialize(transform->serialize());

  auto new_output = new_transform->applyStateless(columns);
  auto new_column = original_output.getArrayColumn<uint32_t>("encoded");

  ASSERT_EQ(original_output.numRows(), new_output.numRows());

  for (size_t i = 0; i < original_output.numRows(); i++) {
    auto original_row = original_column->row(i);
    auto new_row = new_column->row(i);
    ASSERT_EQ(std::vector<uint32_t>(original_row.begin(), original_row.end()),
              std::vector<uint32_t>(new_row.begin(), new_row.end()));
  }
}

TEST(EncodePositionTest, HashedPositionSerialization) {
  testEncodePositionSerialization(/* hashed_position= */ true);
}

TEST(EncodePositionTest, OffsetPositionSerialization) {
  testEncodePositionSerialization(/* hashed_position= */ false);
}

}  // namespace thirdai::data::tests