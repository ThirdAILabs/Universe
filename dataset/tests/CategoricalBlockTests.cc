#include "BlockTest.h"
#include <gtest/gtest.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/encodings/categorical/StringToUidMap.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <sys/types.h>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

class CategoricalBlockTest : public BlockTest {
 public:
  /**
   * Generates a 2 dimensional matrix of integers in the form of a vector
   * of vectors of integers.
   */
  static std::vector<std::vector<uint32_t>> generate_int_matrix(
      uint32_t n_rows, uint32_t n_cols,
      uint32_t max = std::numeric_limits<uint32_t>::max()) {
    std::vector<std::vector<uint32_t>> matrix;
    for (uint32_t row_idx = 0; row_idx < n_rows; row_idx++) {
      std::vector<uint32_t> row;
      for (uint32_t col = 0; col < n_cols; col++) {
        row.push_back(std::rand() % max);
      }
      matrix.push_back(row);
    }
    return matrix;
  }

  /**
   * Generates a 2 dimensional matrix of strings based on a 2 dimensional
   * matrix of integers. Each inner vector mimics the input format
   * expected by a categorical block.
   */
  static std::vector<std::vector<std::string>> generate_input_matrix(
      std::vector<std::vector<uint32_t>>& int_matrix) {
    std::vector<std::vector<std::string>> str_matrix;
    for (const auto& row : int_matrix) {
      std::vector<std::string> str_row;
      str_row.reserve(row.size());
      for (const auto& col : row) {
        str_row.push_back(std::to_string(col));
      }
      str_matrix.push_back(str_row);
    }
    return str_matrix;
  }
};

/**
 * Tests that categorical block properly encodes numerical IDs given
 * as strings.
 *
 * We don't use a mock categorical encoding model or check
 * ContiguousNumericId separately because the ContiguousNumericId
 * is about as simple as an encoding model gets and any mock encoding
 * we write is mostly going to be the same as ContiguousNumericId.
 */
TEST_F(CategoricalBlockTest, ProducesCorrectVectorsDifferentColumns) {
  std::vector<SegmentedSparseFeatureVector> vecs;
  std::vector<uint32_t> dims{100, 1000, 55};
  std::vector<CategoricalBlock> blocks{
      {0, dims[0]}, {1, dims[1]}, {2, dims[2]}};

  auto int_matrix = generate_int_matrix(1000, 3);
  auto input_matrix = generate_input_matrix(int_matrix);

  // Encode the input matrix
  for (const auto& row : input_matrix) {
    SegmentedSparseFeatureVector vec;
    for (auto& block : blocks) {
      addVectorSegmentWithBlock(block, row, vec);
    }
    vecs.push_back(std::move(vec));
  }

  // Check that encoded features match.
  ASSERT_EQ(input_matrix.size(), vecs.size());
  for (uint32_t line = 0; line < input_matrix.size(); line++) {
    // Collect expected key value pairs for this line.
    std::unordered_map<uint32_t, float> expected_key_value_pairs;
    uint32_t current_start_idx = 0;
    for (uint32_t col = 0; col < dims.size(); col++) {
      uint32_t dim = dims[col];
      uint32_t idx = current_start_idx + (int_matrix[line][col] % dim);
      expected_key_value_pairs[idx] = 1.0;
      // Since we are composing features, the starting dimension of the next set
      // of categorical features is offset by the dimension of the current
      // categorical features.
      current_start_idx += dim;
    }

    // Collect actual key val pairs. We need to aggregate it in a map because
    // there may be duplicate entries with the same key.
    auto actual_key_value_pairs = vectorEntries(vecs[line]);

    // Check that actual equals expected.
    ASSERT_EQ(actual_key_value_pairs.size(), expected_key_value_pairs.size());
    for (const auto& [key, val] : expected_key_value_pairs) {
      ASSERT_EQ(val, actual_key_value_pairs[key]);
    }
  }
}

TEST_F(CategoricalBlockTest, ContiguousNumericIdWithGraph) {
  uint32_t n_ids = 20;
  size_t graph_max_n_neighbors = 10;
  size_t block_max_n_neighbors = 8;

  auto graph = buildGraph(n_ids, graph_max_n_neighbors);
  std::vector<SegmentedSparseFeatureVector> vecs;
  CategoricalBlock block(/* col = */ 0, /* dim = */ n_ids, graph,
                         block_max_n_neighbors);

  auto int_matrix = generate_int_matrix(/* n_rows = */ 1000, /* n_cols = */ 1,
                                        /* max = */ n_ids);
  auto input_matrix = generate_input_matrix(int_matrix);

  for (const auto& row : input_matrix) {
    SegmentedSparseFeatureVector vec;
    addVectorSegmentWithBlock(block, row, vec);
    vecs.push_back(std::move(vec));
  }

  for (auto& vec : vecs) {
    auto entries = vectorEntries(vec);
    uint32_t entries_under_n_id = 0;
    uint32_t current_id = 0;
    for (const auto& [idx, val] : entries) {
      if (idx < n_ids) {
        entries_under_n_id++;
        current_id = idx;
      }
      ASSERT_EQ(val, 1.0);
    }
    ASSERT_EQ(entries_under_n_id, 1);
    auto nbrs = getIntNeighbors(current_id, *graph);
    ASSERT_EQ(entries.size(), std::min(nbrs.size(), block_max_n_neighbors) + 1);

    for (uint32_t i = 0; i < std::min(nbrs.size(), block_max_n_neighbors);
         i++) {
      auto nbr = nbrs[i];
      ASSERT_EQ(entries[nbr + n_ids], 1.0);
    }
  }
}

TEST_F(CategoricalBlockTest, StringToUidMapWithGraph) {
  uint32_t n_ids = 20;
  size_t graph_max_n_neighbors = 10;
  size_t block_max_n_neighbors = 8;

  auto graph = buildGraph(n_ids, graph_max_n_neighbors);
  std::vector<SegmentedSparseFeatureVector> vecs;
  auto map_encoding = std::make_shared<StringToUidMap>(/* n_classes = */ n_ids);
  CategoricalBlock block(/* col = */ 0, map_encoding, graph,
                         block_max_n_neighbors);

  auto int_matrix = generate_int_matrix(/* n_rows = */ 1, /* n_cols = */ 1,
                                        /* max = */ n_ids);
  auto input_matrix = generate_input_matrix(int_matrix);

  for (const auto& row : input_matrix) {
    SegmentedSparseFeatureVector vec;
    addVectorSegmentWithBlock(block, row, vec);
    vecs.push_back(std::move(vec));
  }

  for (auto& vec : vecs) {
    auto entries = vectorEntries(vec);
    uint32_t entries_under_n_id = 0;
    uint32_t current_id = 0;
    for (const auto& [idx, val] : entries) {
      ASSERT_LT(idx, block.featureDim());
      if (idx < n_ids) {
        entries_under_n_id++;
        current_id = idx;
      }
      ASSERT_EQ(val, 1.0);
    }
    ASSERT_EQ(entries_under_n_id, 1);

    auto nbrs = (*graph)[map_encoding->uidToClass(current_id)];

    std::unordered_map<std::string, bool> neighbor_exists;
    for (const auto& nbr : nbrs) {
      neighbor_exists[nbr] = true;
    }
    ASSERT_EQ(entries.size(), std::min(nbrs.size(), block_max_n_neighbors) + 1);
    for (const auto& [key, val] : entries) {
      ASSERT_LT(key, block.featureDim());
      if (key > n_ids) {
        ASSERT_EQ(val, 1.0);
        auto class_name = map_encoding->uidToClass(key - n_ids);
        ASSERT_TRUE(neighbor_exists[class_name]);
      }
    }
  }
}

}  // namespace thirdai::dataset
