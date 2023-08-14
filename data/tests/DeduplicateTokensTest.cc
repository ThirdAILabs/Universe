#include <gtest/gtest.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/DeduplicateTokens.h>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::data::tests {

static std::unordered_map<uint32_t, float> indexValuePairs(
    RowView<uint32_t> indices, RowView<float> values) {
  std::unordered_map<uint32_t, float> features;
  for (uint32_t i = 0; i < indices.size(); i++) {
    // Assign instead of add since we assume each index is unique.
    features[indices[i]] = values[i];
  }
  return features;
}

TEST(DeduplicateTokensTest, InputIndicesOnly) {
  auto indices = ArrayColumn<uint32_t>::make({{1, 2, 1, 1}, {1, 2, 3, 4}}, 5);
  ColumnMap columns({{"indices", indices}});
  DeduplicateTokens deduplicate(/* input_indices_column= */ "indices",
                                /* input_values_column= */ std::nullopt,
                                /* output_indices_column= */ "deduped_indices",
                                /* output_values_column= */ "deduped_values");
  columns = deduplicate.applyStateless(columns);
  auto deduped_indices = columns.getArrayColumn<uint32_t>("deduped_indices");
  auto deduped_values = columns.getArrayColumn<float>("deduped_values");

  auto features_row_0 =
      indexValuePairs(deduped_indices->row(0), deduped_values->row(0));
  ASSERT_EQ(features_row_0.size(), 2);
  ASSERT_EQ(features_row_0[1], 3.0);
  ASSERT_EQ(features_row_0[2], 1.0);

  auto features_row_1 =
      indexValuePairs(deduped_indices->row(1), deduped_values->row(1));
  ASSERT_EQ(features_row_1.size(), 4);
  ASSERT_EQ(features_row_1[1], 1.0);
  ASSERT_EQ(features_row_1[2], 1.0);
  ASSERT_EQ(features_row_1[3], 1.0);
  ASSERT_EQ(features_row_1[4], 1.0);

  ASSERT_EQ(deduped_indices->dim().value(), 5);
  ASSERT_EQ(deduped_values->dim(), std::nullopt);
}

TEST(DeduplicateTokensTest, InputIndicesAndValues) {
  auto indices = ArrayColumn<uint32_t>::make({{1, 2, 1, 1}, {1, 2, 3, 4}}, 5);
  auto values =
      ArrayColumn<float>::make({{1, 2, 1, 3}, {1, 2, 3, 4}}, std::nullopt);
  ColumnMap columns({{"indices", indices}, {"values", values}});
  DeduplicateTokens deduplicate(/* input_indices_column= */ "indices",
                                /* input_values_column= */ "values",
                                /* output_indices_column= */ "deduped_indices",
                                /* output_values_column= */ "deduped_values");
  columns = deduplicate.applyStateless(columns);
  auto deduped_indices = columns.getArrayColumn<uint32_t>("deduped_indices");
  auto deduped_values = columns.getArrayColumn<float>("deduped_values");

  auto features_row_0 =
      indexValuePairs(deduped_indices->row(0), deduped_values->row(0));
  ASSERT_EQ(features_row_0.size(), 2);
  ASSERT_EQ(features_row_0[1], 5);
  ASSERT_EQ(features_row_0[2], 2);

  auto features_row_1 =
      indexValuePairs(deduped_indices->row(1), deduped_values->row(1));
  ASSERT_EQ(features_row_1.size(), 4);
  ASSERT_EQ(features_row_1[1], 1);
  ASSERT_EQ(features_row_1[2], 2);
  ASSERT_EQ(features_row_1[3], 3);
  ASSERT_EQ(features_row_1[4], 4);

  ASSERT_EQ(deduped_indices->dim().value(), 5);
  ASSERT_EQ(deduped_values->dim(), std::nullopt);
}

}  // namespace thirdai::data::tests