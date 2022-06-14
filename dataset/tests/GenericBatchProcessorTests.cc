#include "MockBlock.h"
#include <gtest/gtest.h>
#include <dataset/src/bolt_datasets/batch_processors/GenericBatchProcessor.h>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace thirdai::dataset {

std::vector<std::vector<float>> makeFloatMatrix(uint32_t n_rows,
                                                uint32_t n_cols) {
  assert(n_cols >= 1);
  std::vector<std::vector<float>> matrix(n_rows, std::vector<float>(n_cols));

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::normal_distribution<float> dist(0.0, 1.0);

  for (auto& row : matrix) {
    std::generate(row.begin(), row.end(), [&]() { return dist(eng); });
  }

  return matrix;
}

std::vector<std::string> makeStringMatrix(
    std::vector<std::vector<float>>& float_matrix) {
  std::vector<std::string> matrix(float_matrix.size());

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::normal_distribution<float> dist(0.0, 1.0);

  for (uint32_t i = 0; i < float_matrix.size(); i++) {
    std::stringstream ss;
    // Set precision to a high number so precision is preserved
    // after processing.
    ss << std::setprecision(30) << float_matrix[i][0];
    for (uint32_t col = 1; col < float_matrix[i].size(); col++) {
      ss << "," << std::setprecision(30) << float_matrix[i][col];
    }
    matrix[i] = ss.str();
  }

  return matrix;
}

std::vector<std::shared_ptr<Block>> makeMockBlocks(
    std::vector<bool> dense_configs) {
  std::vector<std::shared_ptr<Block>> blocks;
  for (uint32_t i = 0; i < dense_configs.size(); i++) {
    blocks.push_back(std::make_shared<MockBlock>(
        /* column = */ i, /* dense = */ dense_configs[i]));
  }
  return blocks;
}

void checkMatrixAndProcessedBatchEquality(
    std::vector<std::vector<float>>& matrix,
    std::optional<BoltDataLabelPair<bolt::BoltBatch>>& processed,
    bool expect_input_dense, bool expect_label_dense) {
  for (uint32_t i = 0; i < matrix.size(); i++) {
    for (uint32_t j = 0; j < matrix[i].size(); j++) {
      ASSERT_EQ(expect_input_dense, processed->first[i].isDense());
      ASSERT_EQ(expect_label_dense, processed->second[i].isDense());

      ASSERT_FLOAT_EQ(matrix[i][j], processed->first[i].activations[j]);
      ASSERT_FLOAT_EQ(matrix[i][j], processed->second[i].activations[j]);

      if (!processed->first[i].isDense()) {
        ASSERT_EQ(processed->first[i].active_neurons[j], j);
      }
      if (!processed->second[i].isDense()) {
        ASSERT_EQ(processed->second[i].active_neurons[j], j);
      }
    }
  }
}

TEST(GenericBatchProcessorTest, DenseInputDenseLabel) {
  auto float_batch = makeFloatMatrix(/* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeStringMatrix(float_batch);
  GenericBatchProcessor processor(makeMockBlocks({true, true, true}),
                                  makeMockBlocks({true, true, true}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ true,
                                       /* expect_label_dense = */ true);
}

TEST(GenericBatchProcessorTest, SparseInputDenseLabel) {
  auto float_batch = makeFloatMatrix(/* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeStringMatrix(float_batch);
  GenericBatchProcessor processor(makeMockBlocks({false, false, false}),
                                  makeMockBlocks({true, true, true}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ false,
                                       /* expect_label_dense = */ true);
}

TEST(GenericBatchProcessorTest, SparseInputSparseLabel) {
  auto float_batch = makeFloatMatrix(/* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeStringMatrix(float_batch);
  GenericBatchProcessor processor(makeMockBlocks({false, false, false}),
                                  makeMockBlocks({false, false, false}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ false,
                                       /* expect_label_dense = */ false);
}

TEST(GenericBatchProcessorTest, DenseInputSparseLabel) {
  auto float_batch = makeFloatMatrix(/* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeStringMatrix(float_batch);
  GenericBatchProcessor processor(makeMockBlocks({true, true, true}),
                                  makeMockBlocks({false, false, false}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ true,
                                       /* expect_label_dense = */ false);
}

TEST(GenericBatchProcessorTest, Mix) {
  auto float_batch = makeFloatMatrix(/* n_rows = */ 256, /* n_cols = */ 3);
  auto string_batch = makeStringMatrix(float_batch);
  GenericBatchProcessor processor(makeMockBlocks({true, false, true}),
                                  makeMockBlocks({false, true, false}));
  auto processed_batch = processor.createBatch(string_batch);
  checkMatrixAndProcessedBatchEquality(float_batch, processed_batch,
                                       /* expect_input_dense = */ false,
                                       /* expect_label_dense = */ false);
}

}  // namespace thirdai::dataset