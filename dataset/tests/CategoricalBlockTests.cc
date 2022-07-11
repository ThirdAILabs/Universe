#include <gtest/gtest.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <sys/types.h>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

class CategoricalBlockTest : public testing::Test {
 public:
  /**
   * Generates a 2 dimensional matrix of integers in the form of a vector
   * of vectors of integers.
   */
  static std::vector<std::vector<uint32_t>> generate_int_matrix(
      uint32_t n_rows, uint32_t n_cols) {
    std::vector<std::vector<uint32_t>> matrix;
    for (uint32_t row_idx = 0; row_idx < n_rows; row_idx++) {
      std::vector<uint32_t> row;
      for (uint32_t col = 0; col < n_cols; col++) {
        row.push_back(std::rand());
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

  /**
   * Helper function to access addVectorSegment() method of CategoricalBlock,
   * which is private.
   */
  static void addVectorSegmentWithBlock(
      CategoricalBlock& block, const std::vector<std::string>& input_row,
      SegmentedSparseFeatureVector& vec, std::string& block_exception_message) {
    std::vector<std::string_view> input_row_view(input_row.size());
    for (uint32_t i = 0; i < input_row.size(); i++) {
      input_row_view[i] =
          std::string_view(input_row[i].c_str(), input_row[i].size());
    }
    block.addVectorSegment(input_row_view, vec, block_exception_message);
  }

  /**
   * Helper function to access entries() method of SegmentedFeatureVector,
   * which is private.
   */
  static std::unordered_map<uint32_t, float> vectorEntries(
      SegmentedFeatureVector& vec) {
    return vec.entries();
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
  std::string block_exception_message;
  for (const auto& row : input_matrix) {
    SegmentedSparseFeatureVector vec;
    for (auto& block : blocks) {
      addVectorSegmentWithBlock(block, row, vec, block_exception_message);
    }
    vecs.push_back(std::move(vec));
  }
  if (!block_exception_message.empty()) {
    throw std::invalid_argument(block_exception_message);
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

}  // namespace thirdai::dataset
