#include <gtest/gtest.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <memory>
#include <vector>

namespace thirdai::dataset {

class BlockTest : public testing::Test {
 public:
  using StringMatrix = std::vector<std::vector<std::string>>;

  /**
   * Builds sparse segmented vectors according to the supplied
   * string matrix and feature blocks.
   */
  static std::vector<SegmentedSparseFeatureVector> makeSparseSegmentedVecs(
      StringMatrix& matrix, std::vector<std::shared_ptr<Block>>& blocks) {
    std::vector<SegmentedSparseFeatureVector> vecs;
    for (const auto& row : matrix) {
      SegmentedSparseFeatureVector vec;
      for (auto& block : blocks) {
        addVectorSegmentWithBlock(*block, row, vec);
      }
      vecs.push_back(std::move(vec));
    }
    return vecs;
  }

  /**
   * Builds dense segmented vectors according to the supplied
   * string matrix and feature blocks.
   */
  static std::vector<SegmentedDenseFeatureVector> makeDenseSegmentedVecs(
      StringMatrix& matrix, std::vector<std::shared_ptr<Block>>& blocks) {
    std::vector<SegmentedDenseFeatureVector> vecs;
    for (const auto& row : matrix) {
      SegmentedDenseFeatureVector vec;
      for (auto& block : blocks) {
        addVectorSegmentWithBlock(*block, row, vec);
      }
      vecs.push_back(std::move(vec));
    }
    return vecs;
  }

  /**
   * Helper function to access extendVector() method of TextBlock,
   * which is private.
   */
  static void addVectorSegmentWithBlock(
      Block& block, const std::vector<std::string>& input_row,
      SegmentedFeatureVector& vec) {
    std::vector<std::string_view> input_row_view(input_row.size());
    for (uint32_t i = 0; i < input_row.size(); i++) {
      input_row_view[i] =
          std::string_view(input_row[i].c_str(), input_row[i].size());
    }
    block.addVectorSegment(input_row_view, vec);
  }

  /**
   * Helper function to access entries() method of ExtendableVector,
   * which is private.
   */
  static std::unordered_map<uint32_t, float> vectorEntries(
      SegmentedFeatureVector& vec) {
    return vec.entries();
  }

  static uint32_t sumMapValues(std::unordered_map<uint32_t, float>& map) {
    float sum = 0;
    for (const auto [_, v] : map) {
      sum += v;
    }
    return static_cast<uint32_t>(sum);
  }
};

}  // namespace thirdai::dataset