#include <gtest/gtest.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/RegressionBinning.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::data::tests {

void testRegressionBinning(const Transformation& binning) {
  ColumnMap columns(
      {{"input", ValueColumn<float>::make({2, 11, 13, 17, 31, 42, 99})}});

  columns = binning.applyStateless(columns);

  std::vector<std::vector<uint32_t>> expected_bins = {
      {0, 1, 2, 3},          {0, 1, 2, 3},    {0, 1, 2, 3}, {0, 1, 2, 3, 4},
      {2, 3, 4, 5, 6, 7, 8}, {5, 6, 7, 8, 9}, {6, 7, 8, 9},
  };

  ASSERT_EQ(columns.numRows(), expected_bins.size());

  auto bins = columns.getArrayColumn<uint32_t>("output");

  for (size_t i = 0; i < expected_bins.size(); i++) {
    auto row = bins->row(i);
    ASSERT_EQ(std::vector<uint32_t>(row.begin(), row.end()), expected_bins[i]);
  }
}

TEST(RegressionBinningTest, CorrectLabels) {
  RegressionBinning binning("input", "output", /* min= */ 10, /* max= */ 50,
                            /* num_bins= */ 10, /* correct_label_radius= */ 3);

  testRegressionBinning(binning);
}

TEST(RegressionBinningTest, Serialization) {
  RegressionBinning binning("input", "output", /* min= */ 10, /* max= */ 50,
                            /* num_bins= */ 10, /* correct_label_radius= */ 3);

  auto transformation = Transformation::deserialize(binning.serialize());

  testRegressionBinning(*transformation);
}

}  // namespace thirdai::data::tests