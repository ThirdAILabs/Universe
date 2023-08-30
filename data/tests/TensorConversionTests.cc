#include "gtest/gtest.h"
#include <bolt_vector/tests/BoltVectorTestUtils.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <numeric>
#include <optional>

namespace thirdai::data::tests {

void runConversionTest(bool specify_values, bool values_sum_to_one = true) {
  std::vector<std::vector<uint32_t>> indices;
  std::vector<std::vector<float>> values;

  std::vector<size_t> row_lens = {4, 7, 6, 4, 12, 15, 8, 3};

  size_t value = 0;
  for (size_t len : row_lens) {
    std::vector<uint32_t> row_indices(len);
    std::iota(row_indices.begin(), row_indices.end(), value);
    indices.push_back(row_indices);

    std::vector<float> row_values(len);
    std::iota(row_values.begin(), row_values.end(), static_cast<float>(value));
    values.push_back(row_values);

    value += len;
  }

  auto indices_copy = indices;
  auto values_copy = values;

  auto indices_col =
      ArrayColumn<uint32_t>::make(std::move(indices_copy), /* dim= */ 1000);
  auto values_col =
      ArrayColumn<float>::make(std::move(values_copy), /* dim= */ std::nullopt);

  ColumnMap columns({{"indices", indices_col}, {"values", values_col}});

  ValueFillType fill_type =
      values_sum_to_one ? ValueFillType::SumToOne : ValueFillType::Ones;
  OutputColumns to_convert = specify_values
                                 ? OutputColumns("indices", "values")
                                 : OutputColumns("indices", fill_type);

  auto tensors = toTensorBatches(columns, {to_convert}, /* batch_size= */ 3);

  size_t row_cnt = 0;
  size_t value_cnt = 0;
  for (const auto& batch : tensors) {
    ASSERT_EQ(batch.size(), 1);
    ASSERT_EQ(batch[0]->dim(), 1000);
    ASSERT_FALSE(batch[0]->nonzeros().has_value());
    ASSERT_GT(batch[0]->batchSize(), 0);

    for (size_t i = 0; i < batch[0]->batchSize(); i++) {
      const BoltVector& vec = batch[0]->getVector(i);

      EXPECT_EQ(indices.at(row_cnt).size(), vec.len);

      for (size_t j = 0; j < vec.len; j++) {
        EXPECT_EQ(vec.active_neurons[j], value_cnt);
        if (specify_values) {
          EXPECT_EQ(vec.activations[j], static_cast<float>(value_cnt));
        } else {
          if (values_sum_to_one) {
            EXPECT_FLOAT_EQ(vec.activations[i],
                            1.0 / indices.at(row_cnt).size());
          } else {
            EXPECT_EQ(vec.activations[i], 1.0);
          }
        }
        value_cnt++;
      }
      row_cnt++;
    }
  }

  ASSERT_EQ(row_cnt, row_lens.size());
}

TEST(TensorConversionTests, WithValues) {
  runConversionTest(/* specify_values= */ true);
}

TEST(TensorConversionTests, FillValuesOnes) {
  runConversionTest(/* specify_values= */ false,
                    /* values_sum_to_one= */ false);
}

TEST(TensorConversionTests, FillValuesSumToOne) {
  runConversionTest(/* specify_values= */ false, /* values_sum_to_one= */ true);
}

using thirdai::tests::BoltVectorTestUtils;

TEST(TensorConversionTests, MultipleOutputTensorsPerRow) {
  auto indices_1 =
      ArrayColumn<uint32_t>::make({{0, 1, 2}, {3, 4}, {5, 6, 7}}, /* dim= */ 8);
  auto values_1 = ArrayColumn<float>::make(
      {{0.25, 1.25, 2.25}, {3.25, 4.25}, {5.25, 6.25, 7.25}},
      /* dim= */ std::nullopt);

  auto indices_2 = ArrayColumn<uint32_t>::make({{10, 20}, {30, 40}, {50, 60}},
                                               /* dim= */ 100);

  ColumnMap columns({{"indices_1", indices_1},
                     {"values_1", values_1},
                     {"indices_2", indices_2}});

  auto tensors = toTensorBatches(
      columns,
      {OutputColumns("indices_1", "values_1"), OutputColumns("indices_2")},
      /* batch_size= */ 2);

  ASSERT_EQ(tensors.size(), 2);
  ASSERT_EQ(tensors.at(0).size(), 2);
  ASSERT_EQ(tensors.at(1).size(), 2);

  ASSERT_EQ(tensors.at(0).at(0)->batchSize(), 2);
  ASSERT_EQ(tensors.at(0).at(1)->batchSize(), 2);
  ASSERT_EQ(tensors.at(1).at(0)->batchSize(), 1);
  ASSERT_EQ(tensors.at(1).at(1)->batchSize(), 1);

  BoltVectorTestUtils::assertBoltVectorsAreEqual(
      tensors.at(0).at(0)->getVector(0),
      BoltVector::makeSparseVector({0, 1, 2}, {0.25, 1.25, 2.25}));
  BoltVectorTestUtils::assertBoltVectorsAreEqual(
      tensors.at(0).at(0)->getVector(1),
      BoltVector::makeSparseVector({3, 4}, {3.25, 4.25}));
  BoltVectorTestUtils::assertBoltVectorsAreEqual(
      tensors.at(1).at(0)->getVector(0),
      BoltVector::makeSparseVector({5, 6, 7}, {5.25, 6.25, 7.25}));

  BoltVectorTestUtils::assertBoltVectorsAreEqual(
      tensors.at(0).at(1)->getVector(0),
      BoltVector::makeSparseVector({10, 20}, {1.0, 1.0}));
  BoltVectorTestUtils::assertBoltVectorsAreEqual(
      tensors.at(0).at(1)->getVector(1),
      BoltVector::makeSparseVector({30, 40}, {1.0, 1.0}));
  BoltVectorTestUtils::assertBoltVectorsAreEqual(
      tensors.at(1).at(1)->getVector(0),
      BoltVector::makeSparseVector({50, 60}, {1.0, 1.0}));
}

}  // namespace thirdai::data::tests