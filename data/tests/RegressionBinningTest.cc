#include <gtest/gtest.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/RegressionBinning.h>

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

  // We down cast to transformation because otherwise it was trying to call
  // the cereal "serialize" method. This can be removed once cereal is
  // officially depreciated.
  auto transformation = Transformation::deserialize(
      dynamic_cast<Transformation*>(&binning)->serialize());

  testRegressionBinning(*transformation);
}

}  // namespace thirdai::data::tests