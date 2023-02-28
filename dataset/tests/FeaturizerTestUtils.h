#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/tests/MockBlock.h>
#include <iomanip>
#include <memory>
#include <random>
#include <vector>
namespace thirdai::dataset {

class FeaturizerTestUtils {
 public:
  /**
   * Helper function to generate random matrix of floating point
   * numbers with the specified number rows and columns.
   */
  static std::vector<std::vector<float>> makeRandomDenseMatrix(
      uint32_t n_rows, uint32_t n_cols) {
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

  /**
   * Helper function that generates a BlockList of mock blocks
   * according to the given density configurations.
   * E.g. makeMockBlocks({true, false, true}) creates a
   * vector consisting of a dense block, followed by a
   * sparse block, followed by another dense block.
   */
  static dataset::BlockList makeMockBlocks(std::vector<bool> dense_configs) {
    std::vector<std::shared_ptr<Block>> blocks;
    for (uint32_t i = 0; i < dense_configs.size(); i++) {
      blocks.push_back(std::make_shared<MockBlock>(
          /* column = */ i, /* dense = */ dense_configs[i]));
    }
    return dataset::BlockList(std::move(blocks));
  }

  static std::vector<std::string> floatVecToStringVec(
      const std::vector<float>& row) {
    std::vector<std::string> string_vec;
    for (const auto& elem : row) {
      std::stringstream val_ss;
      // Set precision to a high number so precision is preserved
      // after processing.
      val_ss << std::setprecision(30) << elem;
      string_vec.push_back(val_ss.str());
    }
    return string_vec;
  }

  static std::vector<BoltVector> columnInOutputBatch(
      const std::vector<std::vector<BoltVector>>& batch, uint32_t column_id) {
    std::vector<BoltVector> column;
    column.reserve(batch.size());
    for (const auto& row : batch) {
      column.push_back(row[column_id]);
    }
    return column;
  }
};

}  // namespace thirdai::dataset