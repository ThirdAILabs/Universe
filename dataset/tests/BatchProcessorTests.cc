#include <bolt/src/layers/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <dataset/src/core/BatchProcessor.h>
#include <sys/types.h>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

/**
 * A mock block that parses a floating point number
 * in the specified column and produces a one-dimensional
 * vector with the number as its value.
 */
class MockBlock : public Block {
 public:
  explicit MockBlock(uint32_t column, bool dense)
      : _column(column), _dense(dense) {}

  uint32_t featureDim() const override { return 1; };

  bool isDense() const override { return _dense; };

 protected:
  void buildExtension(const std::vector<std::string>& input_row,
                      ExtendableVector& vec) override {
    const std::string& col_str = input_row.at(_column);
    char* end;
    float val = std::strtof(col_str.c_str(), &end);

    if (_dense) {
      vec.addExtensionDenseFeature(val);
    } else {
      vec.addExtensionSparseFeature(0, val);
    }
  };

 private:
  uint32_t _column;
  bool _dense;
};

/**
 * Helper function to generate random matrix of floating point
 * numbers in the range [0, 1) with the specified number rows and columns.
 */
std::vector<std::vector<float>> makeRandomDenseMatrix(size_t n_rows,
                                                      size_t n_cols) {
  // Not the most efficient representation of a matrix but
  // makes the test more readable.
  std::vector<std::vector<float>> matrix(n_rows);
  for (auto& row : matrix) {
    row.resize(n_cols);
    for (auto& elem : row) {
      elem = static_cast<float>(std::rand() & 8) / 256;
    }
  }
  return matrix;
}

/**
 * Helper function to convert each element in a dense
 * matrix into the corresponding string. This mimics the
 * parsed CSV string representation.
 */
std::vector<std::vector<std::string>> makeStringMatrix(
    std::vector<std::vector<float>>& matrix) {
  std::vector<std::vector<std::string>> string_matrix;
  for (const auto& row : matrix) {
    std::vector<std::string> string_row;
    // The reserve below is an unnecessary optimization given that this is just
    // a test but linter gets angry otherwise
    string_row.reserve(row.size());
    for (const auto& elem : row) {
      string_row.push_back(std::to_string(elem));
    }
    string_matrix.push_back(std::move(string_row));
  }
  return string_matrix;
}

/**
 * Helper function to convert a dense matrix into the
 * corresponding dense bolt vector representations
 */
std::vector<bolt::BoltVector> makeDenseBoltVectors(
    std::vector<std::vector<float>>& matrix) {
  std::vector<bolt::BoltVector> vectors;
  // The reserve below is an unnecessary optimization given that this is just a
  // test but linter gets angry otherwise
  vectors.reserve(matrix.size());
  for (const auto& row : matrix) {
    vectors.push_back(bolt::BoltVector::makeDenseVector(row));
  }
  return vectors;
}

/**
 * Helper function to convert a dense matrix into the
 * corresponding sparse bolt vector representations
 */
std::vector<bolt::BoltVector> makeSparseBoltVectors(
    std::vector<std::vector<float>>& matrix) {
  // Make indices for sparse vector (same for every row)
  std::vector<uint32_t> indices(matrix.front().size());
  for (uint32_t i = 0; i < indices.size(); i++) {
    indices[i] = i;
  }

  // Make the bolt vectors.
  std::vector<bolt::BoltVector> vectors;
  // The reserve below is an unnecessary optimization given that this is just a
  // test but linter gets angry otherwise
  vectors.reserve(matrix.size());
  for (const auto& row : matrix) {
    vectors.push_back(bolt::BoltVector::makeSparseVector(indices, row));
  }
  return vectors;
}

/**
 * Helper function that generates n mock blocks
 * If dense is true, all blocks are dense. Otherwise, all blocks sparse.
 * If mixed_dense is true, the first block is sparse, the second is dense,
 * and the last is sparse again.
 */
std::vector<std::shared_ptr<Block>> makeNMockBlocks(uint32_t n_blocks,
                                                    bool dense,
                                                    bool mixed_dense = false) {
  std::vector<std::shared_ptr<Block>> blocks;

  if (!mixed_dense) {
    for (uint32_t i = 0; i < n_blocks; i++) {
      auto mock_block_ptr = std::make_shared<MockBlock>(i, dense);
      blocks.push_back(std::static_pointer_cast<Block>(mock_block_ptr));
    }
    return blocks;
  }

  // Mixed dense case
  for (uint32_t i = 0; i < n_blocks; i++) {
    auto mock_block_ptr = std::make_shared<MockBlock>(i, i == 1);
    blocks.push_back(std::static_pointer_cast<Block>(mock_block_ptr));
  }
  return blocks;
}

/**
 * Helper function that checks that the given dataset contains the same
 * bolt vectors as the concatenation of the two given vectors of bolt vectors
 *
 * If check_labels = true, check the labels of the dataset. Otherwise, check the
 * inputs.
 */
void assertSameVectorsSameOrder(std::vector<bolt::BoltVector> bolt_vecs_1,
                                std::vector<bolt::BoltVector> bolt_vecs_2,
                                BoltDataset& dataset) {
  uint32_t batch_size = dataset.at(0).getBatchSize();
  for (size_t batch_i = 0; batch_i < dataset.numBatches(); batch_i++) {
    for (size_t vec_i = 0; vec_i < dataset.at(batch_i).getBatchSize();
         vec_i++) {
      const auto& dataset_vec = dataset.at(batch_i)[vec_i];

      size_t idx = batch_i * batch_size + vec_i;
      const bolt::BoltVector& vec = idx < bolt_vecs_1.size()
                                        ? bolt_vecs_1[idx]
                                        : bolt_vecs_2[idx - bolt_vecs_1.size()];

      // Assert same length
      ASSERT_EQ(dataset_vec.len, vec.len);

      // Assert same dense-ness
      ASSERT_EQ(dataset_vec.active_neurons == nullptr,
                vec.active_neurons == nullptr);

      // Assert same active neurons if sparse
      if (vec.active_neurons != nullptr) {
        for (uint32_t i = 0; i < vec.len; i++) {
          ASSERT_EQ(dataset_vec.active_neurons[i], vec.active_neurons[i]);
        }
      }

      // Assert same activations
      for (uint32_t i = 0; i < vec.len; i++) {
        ASSERT_EQ(dataset_vec.activations[i], vec.activations[i]);
      }
    }
  }
}

/**
 * Helper function that checks whether a batch processor produces a dataset
 * with the correct bolt vectors, given whether the inputs and targets
 * are dense or sparse.
 *
 * If has_labels is false, label_dense is ignored.
 * If mixed_dense_input_and_label is true, input_dense and label_dense are
 * ignored.
 */
void checkCorrectUnshuffledDatasetImpl(
    bool has_labels, bool input_dense, bool label_dense = false,
    bool mixed_dense_input_and_label = false) {
  // Generate mock data
  size_t n_cols = 3;

  auto dense_matrix_1 = makeRandomDenseMatrix(/* n_rows = */ 500, n_cols);
  auto str_matrix_1 = makeStringMatrix(dense_matrix_1);
  auto dense_bolt_vecs_1 = makeDenseBoltVectors(dense_matrix_1);
  auto sparse_bolt_vecs_1 = makeSparseBoltVectors(dense_matrix_1);

  auto dense_matrix_2 = makeRandomDenseMatrix(/* n_rows = */ 500, n_cols);
  auto str_matrix_2 = makeStringMatrix(dense_matrix_2);
  auto dense_bolt_vecs_2 = makeDenseBoltVectors(dense_matrix_2);
  auto sparse_bolt_vecs_2 = makeSparseBoltVectors(dense_matrix_2);
  uint32_t output_batch_size = 256;

  // Initialize batch processor and process batches
  auto input_blocks =
      makeNMockBlocks(n_cols, input_dense, mixed_dense_input_and_label);
  std::vector<std::shared_ptr<Block>> target_blocks;
  if (has_labels) {
    target_blocks =
        makeNMockBlocks(n_cols, label_dense, mixed_dense_input_and_label);
  }

  BatchProcessor processor(input_blocks, target_blocks, output_batch_size);
  processor.processBatch(str_matrix_1);
  processor.processBatch(str_matrix_2);

  auto [input_dataset_ptr, target_dataset_ptr] =
      processor.exportInMemoryDataset();

  // Assertions
  if (input_dense & !mixed_dense_input_and_label) {
    assertSameVectorsSameOrder(dense_bolt_vecs_1, dense_bolt_vecs_2,
                               *input_dataset_ptr);
  } else {
    assertSameVectorsSameOrder(sparse_bolt_vecs_1, sparse_bolt_vecs_2,
                               *input_dataset_ptr);
  }

  if (has_labels) {
    if (label_dense & !mixed_dense_input_and_label) {
      assertSameVectorsSameOrder(dense_bolt_vecs_1, dense_bolt_vecs_2,
                                 *target_dataset_ptr);
    } else {
      assertSameVectorsSameOrder(sparse_bolt_vecs_1, sparse_bolt_vecs_2,
                                 *target_dataset_ptr);
    }
  } else {
    ASSERT_EQ(target_dataset_ptr, nullptr);
  }
}

/**
 * Checks that the batch processor produces a dataset with the correct
 * bolt vectors, in the correct order.
 *
 * We check 7 cases:
 *  - sparse input, sparse label
 *  - sparse input, dense label
 *  - dense input, sparse label
 *  - dense input, dense label
 *  - mixed sparse and dense input and label
 *  - sparse input, no label
 *  - dense input, no label
 *
 * No shuffling.
 * We don't check the no label case yet because we haven't finalized
 * designing what a batch without labels looks like.
 */
TEST(BatchProcessorTest, ProducesCorrectUnshuffledDataset) {
  // Sparse input, sparse label
  checkCorrectUnshuffledDatasetImpl(/* has_labels = */ true,
                                    /* input_dense = */ false,
                                    /* label_dense = */ false);
  // Sparse input, dense label
  checkCorrectUnshuffledDatasetImpl(/* has_labels = */ true,
                                    /* input_dense = */ false,
                                    /* label_dense = */ true);
  // Dense input, sparse label
  checkCorrectUnshuffledDatasetImpl(/* has_labels = */ true,
                                    /* input_dense = */ true,
                                    /* label_dense = */ false);
  // Dense input, dense label
  checkCorrectUnshuffledDatasetImpl(/* has_labels = */ true,
                                    /* input_dense = */ true,
                                    /* label_dense = */ true);
  // Sparse input, no label
  checkCorrectUnshuffledDatasetImpl(/* has_labels = */ false,
                                    /* input_dense = */ false);
  // Dense input, no label
  checkCorrectUnshuffledDatasetImpl(/* has_labels = */ false,
                                    /* input_dense = */ true);
  // Mixed dense and sparse input and label
  checkCorrectUnshuffledDatasetImpl(/* has_labels = */ true,
                                    /* input_dense = */ true,
                                    /* label_dense = */ true,
                                    /* mixed_dense_input_and_label = */ true);
}

/**
 * Helper function that checks that the given vector of one-dimensional
 * bolt vectors are a permutation of the given vector of floats.
 * Assumes label is always equal to value at 0th position of input.
 */
void checkIsPermutation(const BoltDataset& input_dataset,
                        const BoltDataset& target_dataset,
                        const std::vector<float>& values) {
  // Check that the permutation is valid by checking two things:
  // 1. value at position 1 of label and input are the equal.
  // 2. Both dataset and values have the same number of each value.
  std::unordered_map<float, int32_t> value_counts;
  for (const auto& val : values) {
    value_counts[val]++;
  }

  for (uint32_t batch_i = 0; batch_i < input_dataset.numBatches(); batch_i++) {
    for (uint32_t vec_i = 0; vec_i < input_dataset.at(batch_i).getBatchSize();
         vec_i++) {
      ASSERT_EQ(input_dataset.at(batch_i)[vec_i].activations[0],
                target_dataset.at(batch_i)[vec_i].activations[0]);
      value_counts[input_dataset.at(batch_i)[vec_i].activations[0]] -= 1;
    }
  }

  for (const auto& [val, count] : value_counts) {
    ASSERT_EQ(count, 0);
  }
}

/**
 * Helper function to check the equality of dataset orderings.
 * If assert_equal is true, asserts that the orderings are equal.
 * Otherwise, asserts that orderings are different.
 */
void checkDatasetOrderEquality(const BoltDataset& input_dataset_1,
                               const BoltDataset& input_dataset_2,
                               bool assert_equal) {
  uint32_t n_seen = 0;
  for (uint32_t batch_i = 0; batch_i < input_dataset_1.numBatches();
       batch_i++) {
    for (uint32_t vec_i = 0; vec_i < input_dataset_2.at(batch_i).getBatchSize();
         vec_i++) {
      if (input_dataset_1.at(batch_i)[vec_i].activations[0] !=
          input_dataset_2.at(batch_i)[vec_i].activations[0]) {
        break;
      }
      n_seen++;
    }
  }
  if (assert_equal) {
    ASSERT_EQ(n_seen, input_dataset_1.len());
  } else {
    ASSERT_LT(n_seen, input_dataset_1.len());
  }
}

TEST(BatchProcessorTest, ProducesCorrectShuffledDataset) {
  // Mock dataset is range(0.0, 1000.0);
  uint32_t n_rows = 1000;
  std::vector<float> mock_data_seq(n_rows);
  for (uint32_t i = 0; i < mock_data_seq.size(); i++) {
    mock_data_seq[i] = i;
  }

  // Make string version for input to batch processor.
  std::vector<std::vector<std::string>> mock_data_str;
  mock_data_str.reserve(n_rows);
  for (const auto& elem : mock_data_seq) {
    std::vector<std::string> row;
    row.push_back(std::to_string(elem));
    mock_data_str.push_back(std::move(row));
  }

  // Set up batch processor with one block.
  auto mock_block_ptr =
      std::make_shared<MockBlock>(/* column = */ 0, /* dense = */ true);
  std::vector<std::shared_ptr<Block>> blocks{
      std::static_pointer_cast<Block>(mock_block_ptr)};
  uint32_t output_batch_size = 256;
  BatchProcessor bp(blocks, blocks, output_batch_size);

  // Export unshuffled dataset.
  bp.processBatch(mock_data_str);
  auto [input_unshuf_ptr, target_unshuf_ptr] = bp.exportInMemoryDataset();

  // Export shuffled dataset without seed.
  // Check validity of permutation and that the order is different than
  // ds_unshuffled.
  // Must call processBatch again because batch processor is emptied
  // after exporting.
  bp.processBatch(mock_data_str);
  auto [input_shuf_ptr, target_shuf_ptr] =
      bp.exportInMemoryDataset(/* shuffle = */ true);
  checkIsPermutation(*input_shuf_ptr, *target_shuf_ptr, mock_data_seq);
  checkDatasetOrderEquality(*input_shuf_ptr, *input_unshuf_ptr,
                            /* assert_equal = */ false);

  // Export shuffled dataset without seed again.
  // Check validity of permutation and that the order is different than
  // ds_unshuffled and ds_shuffled
  // Must call processBatch again because batch processor is emptied
  // after exporting.
  bp.processBatch(mock_data_str);
  auto [input_shuf_again_ptr, target_shuf_again_ptr] =
      bp.exportInMemoryDataset(/* shuffle = */ true);
  checkIsPermutation(*input_shuf_again_ptr, *target_shuf_again_ptr,
                     mock_data_seq);
  checkDatasetOrderEquality(*input_shuf_again_ptr, *input_unshuf_ptr,
                            /* assert_equal = */ false);
  checkDatasetOrderEquality(*input_shuf_again_ptr, *input_shuf_ptr,
                            /* assert_equal = */ false);

  // Separately export shuffled dataset with the same seed.
  // Check validity of permutation and that the order is different than
  // ds_unshuffled, and check that the orderings are consistent when
  // seeded with the same number.
  bp.processBatch(mock_data_str);
  auto [input_shuf_seeded_1_ptr, target_shuf_seeded_1_ptr] =
      bp.exportInMemoryDataset(/* shuffle = */ true, /* shuffle_seed = */ 0);
  checkIsPermutation(*input_shuf_seeded_1_ptr, *target_shuf_seeded_1_ptr,
                     mock_data_seq);
  checkDatasetOrderEquality(*input_shuf_seeded_1_ptr, *input_unshuf_ptr,
                            /* assert_equal = */ false);

  bp.processBatch(mock_data_str);
  auto [input_shuf_seeded_2_ptr, target_shuf_seeded_2_ptr] =
      bp.exportInMemoryDataset(/* shuffle = */ true, /* shuffle_seed = */ 0);
  checkIsPermutation(*input_shuf_seeded_2_ptr, *target_shuf_seeded_2_ptr,
                     mock_data_seq);
  checkDatasetOrderEquality(*input_shuf_seeded_2_ptr, *input_unshuf_ptr,
                            /* assert_equal = */ false);

  checkDatasetOrderEquality(*input_shuf_seeded_1_ptr, *input_shuf_seeded_2_ptr,
                            /* assert_equal = */ true);
}

}  // namespace thirdai::dataset