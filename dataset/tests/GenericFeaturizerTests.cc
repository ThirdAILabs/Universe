#include "FeaturizerTestUtils.h"
#include "MockBlock.h"
#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/featurizers/GenericFeaturizer.h>
#include <sstream>
#include <string>

namespace thirdai::dataset {

std::vector<std::string> makeCsvRows(std::vector<std::vector<float>>& matrix) {
  std::vector<std::string> csv_rows(matrix.size());
  for (uint32_t i = 0; i < matrix.size(); i++) {
    const auto& string_row =
        FeaturizerTestUtils::floatVecToStringVec(matrix[i]);
    std::string delim;
    for (const auto& elem : string_row) {
      csv_rows[i] += delim + elem;
      delim = ",";
    }
  }

  return csv_rows;
}

void checkMatrixAndProcessedBatchEquality(
    std::vector<std::vector<float>>& matrix, std::vector<BoltBatch>& processed,
    bool expect_input_dense, bool expect_label_dense) {
  const BoltBatch& input = processed.at(0);
  const BoltBatch& labels = processed.at(1);
  for (uint32_t i = 0; i < matrix.size(); i++) {
    for (uint32_t j = 0; j < matrix[i].size(); j++) {
      ASSERT_EQ(expect_input_dense, input[0].isDense());
      ASSERT_EQ(expect_label_dense, labels[i].isDense());

      ASSERT_FLOAT_EQ(matrix[i][j], input[i].activations[j]);
      ASSERT_FLOAT_EQ(matrix[i][j], labels[i].activations[j]);

      if (!input[i].isDense()) {
        ASSERT_EQ(input[i].active_neurons[j], j);
      }
      if (!labels[i].isDense()) {
        ASSERT_EQ(labels[i].active_neurons[j], j);
      }
    }
  }
}

TEST(GenericFeaturizerTests, DenseInputDenseLabel) {
  auto float_batch = FeaturizerTestUtils::makeRandomDenseMatrix(
      /* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeCsvRows(float_batch);
  GenericFeaturizer processor(
      FeaturizerTestUtils::makeMockBlocks({true, true, true}),
      FeaturizerTestUtils::makeMockBlocks({true, true, true}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ true,
                                       /* expect_label_dense = */ true);
}

TEST(GenericFeaturizerTests, SparseInputDenseLabel) {
  auto float_batch = FeaturizerTestUtils::makeRandomDenseMatrix(
      /* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeCsvRows(float_batch);
  GenericFeaturizer processor(
      FeaturizerTestUtils::makeMockBlocks({false, false, false}),
      FeaturizerTestUtils::makeMockBlocks({true, true, true}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ false,
                                       /* expect_label_dense = */ true);
}

TEST(GenericFeaturizerTests, SparseInputSparseLabel) {
  auto float_batch = FeaturizerTestUtils::makeRandomDenseMatrix(
      /* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeCsvRows(float_batch);
  GenericFeaturizer processor(
      FeaturizerTestUtils::makeMockBlocks({false, false, false}),
      FeaturizerTestUtils::makeMockBlocks({false, false, false}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ false,
                                       /* expect_label_dense = */ false);
}

TEST(GenericFeaturizerTests, DenseInputSparseLabel) {
  auto float_batch = FeaturizerTestUtils::makeRandomDenseMatrix(
      /* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeCsvRows(float_batch);
  GenericFeaturizer processor(
      FeaturizerTestUtils::makeMockBlocks({true, true, true}),
      FeaturizerTestUtils::makeMockBlocks({false, false, false}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ true,
                                       /* expect_label_dense = */ false);
}

TEST(GenericFeaturizerTests, Mix) {
  auto float_batch = FeaturizerTestUtils::makeRandomDenseMatrix(
      /* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeCsvRows(float_batch);
  GenericFeaturizer processor(
      FeaturizerTestUtils::makeMockBlocks({true, false, true}),
      FeaturizerTestUtils::makeMockBlocks({false, true, false}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ false,
                                       /* expect_label_dense = */ false);
}

}  // namespace thirdai::dataset