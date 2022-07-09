#include <gtest/gtest.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <charconv>
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
      StringMatrix& matrix, std::vector<std::shared_ptr<Block>>& blocks,
      size_t batch_interval = 0) {
    std::vector<SegmentedSparseFeatureVector> vecs;
    size_t i = 0;
    for (const auto& row : matrix) {
      i++;
      if (i == batch_interval) {
        for (auto& block : blocks) {
          block->prepareForBatch(toStringViewVec(row));
        }
        i = 0;
      }
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
      StringMatrix& matrix, std::vector<std::shared_ptr<Block>>& blocks,
      size_t batch_interval = 0) {
    std::vector<SegmentedDenseFeatureVector> vecs;
    size_t i = 0;
    for (const auto& row : matrix) {
      i++;
      if (i == batch_interval) {
        for (auto& block : blocks) {
          block->prepareForBatch(toStringViewVec(row));
        }
        i = 0;
      }
      SegmentedDenseFeatureVector vec;
      for (auto& block : blocks) {
        addVectorSegmentWithBlock(*block, row, vec);
      }
      vecs.push_back(std::move(vec));
    }
    return vecs;
  }

  /**
   * Helper function to build vector of string views
   * from vector of strings.
   */
  static std::vector<std::string_view> toStringViewVec(
      const std::vector<std::string>& input_row) {
    std::vector<std::string_view> input_row_view(input_row.size());
    for (uint32_t i = 0; i < input_row.size(); i++) {
      input_row_view[i] =
          std::string_view(input_row[i].c_str(), input_row[i].size());
    }
    return input_row_view;
  }

  /**
   * Helper function to access extendVector() method of TextBlock,
   * which is private.
   */
  static void addVectorSegmentWithBlock(
      Block& block, const std::vector<std::string>& input_row,
      SegmentedFeatureVector& vec) {
    auto input_row_view = toStringViewVec(input_row);
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

  static GraphPtr buildGraph(uint32_t n_ids, uint32_t max_n_neighbors) {
    auto graph = std::make_shared<Graph>();
    for (uint32_t id = 0; id < n_ids; id++) {
      for (uint32_t neighbor = 1; neighbor <= std::min(id, max_n_neighbors);
           neighbor++) {
        uint32_t neighbor_id = (id + neighbor) % n_ids;
        graph->operator[](std::to_string(id))
            .push_back(std::to_string(neighbor_id));
      }
    }
    return graph;
  }

  static std::vector<uint32_t> getIntNeighbors(uint32_t id, Graph& graph) {
    std::vector<uint32_t> int_nbrs;
    for (const auto& str_nbr : graph[std::to_string(id)]) {
      uint32_t nbr_id;
      std::from_chars(str_nbr.data(), str_nbr.data() + str_nbr.size(), nbr_id);
      int_nbrs.push_back(nbr_id);
    }
    return int_nbrs;
  }
};

}  // namespace thirdai::dataset