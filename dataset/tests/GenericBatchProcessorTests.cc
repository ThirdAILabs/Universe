#include "BatchProcessorTestUtils.h"
#include "MockBlock.h"
#include <gtest/gtest.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <sstream>
#include <string>

namespace thirdai::dataset {

std::vector<std::string> makeCsvRows(std::vector<std::vector<float>>& matrix) {
  std::vector<std::string> csv_rows(matrix.size());
  for (uint32_t i = 0; i < matrix.size(); i++) {
    const auto& string_row =
        BatchProcessorTestUtils::floatVecToStringVec(matrix[i]);
    std::string delim;
    for (const auto& elem : string_row) {
      csv_rows[i] += delim + elem;
      delim = ",";
    }
  }

  return csv_rows;
}

void checkMatrixAndProcessedBatchEquality(
    std::vector<std::vector<float>>& matrix,
    std::tuple<bolt::BoltBatch, bolt::BoltBatch>& processed,
    bool expect_input_dense, bool expect_label_dense) {
  for (uint32_t i = 0; i < matrix.size(); i++) {
    for (uint32_t j = 0; j < matrix[i].size(); j++) {
      ASSERT_EQ(expect_input_dense, std::get<0>(processed)[i].isDense());
      ASSERT_EQ(expect_label_dense, std::get<1>(processed)[i].isDense());

      ASSERT_FLOAT_EQ(matrix[i][j], std::get<0>(processed)[i].activations[j]);
      ASSERT_FLOAT_EQ(matrix[i][j], std::get<1>(processed)[i].activations[j]);

      if (!std::get<0>(processed)[i].isDense()) {
        ASSERT_EQ(std::get<0>(processed)[i].active_neurons[j], j);
      }
      if (!std::get<1>(processed)[i].isDense()) {
        ASSERT_EQ(std::get<1>(processed)[i].active_neurons[j], j);
      }
    }
  }
}

TEST(GenericBatchProcessorTests, DenseInputDenseLabel) {
  auto float_batch = BatchProcessorTestUtils::makeRandomDenseMatrix(
      /* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeCsvRows(float_batch);
  GenericBatchProcessor processor(
      BatchProcessorTestUtils::makeMockBlocks({true, true, true}),
      BatchProcessorTestUtils::makeMockBlocks({true, true, true}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ true,
                                       /* expect_label_dense = */ true);
}

TEST(GenericBatchProcessorTests, SparseInputDenseLabel) {
  auto float_batch = BatchProcessorTestUtils::makeRandomDenseMatrix(
      /* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeCsvRows(float_batch);
  GenericBatchProcessor processor(
      BatchProcessorTestUtils::makeMockBlocks({false, false, false}),
      BatchProcessorTestUtils::makeMockBlocks({true, true, true}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ false,
                                       /* expect_label_dense = */ true);
}

TEST(GenericBatchProcessorTests, SparseInputSparseLabel) {
  auto float_batch = BatchProcessorTestUtils::makeRandomDenseMatrix(
      /* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeCsvRows(float_batch);
  GenericBatchProcessor processor(
      BatchProcessorTestUtils::makeMockBlocks({false, false, false}),
      BatchProcessorTestUtils::makeMockBlocks({false, false, false}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ false,
                                       /* expect_label_dense = */ false);
}

TEST(GenericBatchProcessorTests, DenseInputSparseLabel) {
  auto float_batch = BatchProcessorTestUtils::makeRandomDenseMatrix(
      /* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeCsvRows(float_batch);
  GenericBatchProcessor processor(
      BatchProcessorTestUtils::makeMockBlocks({true, true, true}),
      BatchProcessorTestUtils::makeMockBlocks({false, false, false}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ true,
                                       /* expect_label_dense = */ false);
}

TEST(GenericBatchProcessorTests, Mix) {
  auto float_batch = BatchProcessorTestUtils::makeRandomDenseMatrix(
      /* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeCsvRows(float_batch);
  GenericBatchProcessor processor(
      BatchProcessorTestUtils::makeMockBlocks({true, false, true}),
      BatchProcessorTestUtils::makeMockBlocks({false, true, false}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ false,
                                       /* expect_label_dense = */ false);
}

}  // namespace thirdai::dataset