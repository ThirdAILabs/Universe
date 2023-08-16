#include "gtest/gtest.h"
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace thirdai::data::tests {

TEST(PermutationTest, RandomOneToManyPermutation) {
  // Make value column {0, 1, 2, ..., 99}
  std::vector<uint32_t> value_column_data(100);
  std::iota(value_column_data.begin(), value_column_data.end(), 0);
  auto orig_val_col = ValueColumn<uint32_t>::make(std::move(value_column_data),
                                                  /* dim= */ std::nullopt);

  // Make array column {{0, 1}, {1, 2}, ..., {99, 100}}
  std::vector<std::vector<uint32_t>> array_column_data(100);
  for (uint32_t i = 0; i < 100; i++) {
    array_column_data[i] = {i, i + 1};
  }
  auto orig_arr_col = ArrayColumn<uint32_t>::make(std::move(array_column_data),
                                                  /* dim= */ std::nullopt);

  ColumnMap orig_map({{"val", orig_val_col}, {"arr", orig_arr_col}});

  // Make permutation, each position in the original columns map to 10 positions
  // in the permuted column.
  std::vector<size_t> permutation(1000);
  for (uint32_t i = 0; i < permutation.size(); i++) {
    permutation[i] = i / 10;
  }
  std::mt19937 rng(341);
  std::shuffle(permutation.begin(), permutation.end(), rng);

  auto permuted_map = orig_map.permute(permutation);

  auto permuted_val_col = permuted_map.getValueColumn<uint32_t>("val");
  auto permuted_arr_col = permuted_map.getArrayColumn<uint32_t>("arr");

  ASSERT_EQ(permuted_val_col->numRows(), permutation.size());
  ASSERT_EQ(permuted_arr_col->numRows(), permutation.size());

  for (uint32_t i = 0; i < permutation.size(); i++) {
    ASSERT_EQ(permuted_val_col->value(i), orig_val_col->value(permutation[i]));

    ASSERT_EQ(permuted_arr_col->row(i).size(), 2);
    auto expected_array = orig_arr_col->row(permutation[i]);
    ASSERT_EQ(permuted_arr_col->row(i)[0], expected_array[0]);
    ASSERT_EQ(permuted_arr_col->row(i)[1], expected_array[1]);
  }
}

}  // namespace thirdai::data::tests