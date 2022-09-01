#pragma once
#include <dataset/src/blocks/BlockInterface.h>
#include <utility>

namespace thirdai::bolt {

using Blocks = std::vector<std::shared_ptr<dataset::Block>>;

class RootCauseAnalysis {
 public:
  explicit RootCauseAnalysis(Blocks input_blocks)
      : _input_blocks(std::move(input_blocks)) {
    std::vector<uint32_t> offsets;
    offsets.push_back(0);
    for (const auto& block : _input_blocks) {
      offsets.push_back(offsets.back() + block->featureDim());
    }
    _offsets = std::move(offsets);
  }

  std::tuple<std::vector<std::string>, std::vector<float>,
             std::vector<uint32_t>>
  getPercentExplanationWithColumnNames(
      const std::vector<float>& gradients_ratio,
      std::vector<uint32_t> gradients_indices,
      std::unordered_map<uint32_t, std::string> num_to_name) {
    std::vector<std::pair<float, uint32_t>> gradients_ratio_with_indices =
        makeGradientRatiosWithIndicesSorted(gradients_ratio,
                                            std::move(gradients_indices));
    float ratio_sum = 0;
    for (float gradient_ratio : gradients_ratio) {
      ratio_sum += std::abs(gradient_ratio);
    }
    std::vector<std::string> column_names;
    std::vector<float> gradient_percent_ratio;
    std::vector<uint32_t> indices_within_block;
    for (const auto& col : gradients_ratio_with_indices) {
      auto index = getIndexOfBlock(col.second);
      indices_within_block.push_back(col.second - getOffsetAt(index));
      auto column = getColumnNumForBlock(index);
      column_names.push_back(num_to_name[column]);
      gradient_percent_ratio.push_back((col.first / ratio_sum) * 100);
    }
    return std::make_tuple(column_names, gradient_percent_ratio,
                           indices_within_block);
  }

  uint32_t getIndexOfBlock(uint32_t index) {
    auto iter = std::upper_bound(_offsets.begin(), _offsets.end(), index);
    return (iter - _offsets.begin() - 1);
  }

  static std::vector<std::pair<float, uint32_t>>
  makeGradientRatiosWithIndicesSorted(std::vector<float> gradients_ratio,
                                      std::vector<uint32_t> gradients_indices) {
    std::vector<std::pair<float, uint32_t>> gradient_ratios_with_indices;
    for (uint32_t j = 0; j < gradients_ratio.size(); j++) {
      gradient_ratios_with_indices.push_back(
          std::make_pair(gradients_ratio[j], gradients_indices[j]));
    }
    auto func = [](std::pair<float, uint32_t> pair1,
                   std::pair<float, uint32_t> pair2) {
      return abs(pair1.first) > abs(pair2.first);
    };
    sort(gradient_ratios_with_indices.begin(),
         gradient_ratios_with_indices.end(), func);
    return gradient_ratios_with_indices;
  }

  uint32_t getColumnNumForBlock(uint32_t index) {
    return _input_blocks[index]->getColumnNum();
  }

  uint32_t getOffsetAt(uint32_t index) { return _offsets[index]; }

 private:
  Blocks _input_blocks;
  std::vector<uint32_t> _offsets;
};

}  // namespace thirdai::bolt