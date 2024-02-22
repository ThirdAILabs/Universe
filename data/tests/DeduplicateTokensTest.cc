#include <gtest/gtest.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/DeduplicateTokens.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::data::tests {

void compareIndexValues(RowView<uint32_t> indices, RowView<float> values,
                        const std::unordered_map<uint32_t, float>& expected) {
  ASSERT_EQ(indices.size(), expected.size());
  ASSERT_EQ(values.size(), expected.size());
  std::unordered_set<uint32_t> unique_idxs = {indices.begin(), indices.end()};
  ASSERT_EQ(unique_idxs.size(), expected.size());

  for (uint32_t i = 0; i < indices.size(); i++) {
    ASSERT_EQ(expected.at(indices[i]), values[i]);
  }
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

  compareIndexValues(deduped_indices->row(0), deduped_values->row(0),
                     /* expected= */ {{1, 3.0}, {2, 1.0}});
  compareIndexValues(deduped_indices->row(1), deduped_values->row(1),
                     /* expected= */ {{1, 1.0}, {2, 1.0}, {3, 1.0}, {4, 1.0}});

  ASSERT_EQ(deduped_indices->dim().value(), 5);
  ASSERT_EQ(deduped_values->dim(), std::nullopt);
}

void testDeduplicateTokens(const Transformation& transform) {
  auto indices = ArrayColumn<uint32_t>::make({{1, 2, 1, 1}, {1, 2, 3, 4}}, 5);
  auto values =
      ArrayColumn<float>::make({{1, 2, 1, 3}, {1, 2, 3, 4}}, std::nullopt);
  ColumnMap columns({{"indices", indices}, {"values", values}});

  columns = transform.applyStateless(columns);

  auto deduped_indices = columns.getArrayColumn<uint32_t>("deduped_indices");
  auto deduped_values = columns.getArrayColumn<float>("deduped_values");

  compareIndexValues(deduped_indices->row(0), deduped_values->row(0),
                     /* expected= */ {{1, 5.0}, {2, 2.0}});
  compareIndexValues(deduped_indices->row(1), deduped_values->row(1),
                     /* expected= */ {{1, 1.0}, {2, 2.0}, {3, 3.0}, {4, 4.0}});

  ASSERT_EQ(deduped_indices->dim().value(), 5);
  ASSERT_EQ(deduped_values->dim(), std::nullopt);
}

TEST(DeduplicateTokensTest, InputIndicesAndValues) {
  DeduplicateTokens deduplicate(
      /* input_indices_column= */ "indices",
      /* input_values_column= */ "values",
      /* output_indices_column= */ "deduped_indices",
      /* output_values_column= */ "deduped_values");

  testDeduplicateTokens(deduplicate);
}

TEST(DeduplicateTokensTest, Serialization) {
  DeduplicateTokens deduplicate(
      /* input_indices_column= */ "indices",
      /* input_values_column= */ "values",
      /* output_indices_column= */ "deduped_indices",
      /* output_values_column= */ "deduped_values");

  auto new_transformation =
      Transformation::deserialize(deduplicate.serialize());

  testDeduplicateTokens(*new_transformation);
}

}  // namespace thirdai::data::tests