#pragma once
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <utility>

namespace thirdai::bolt {

using Blocks = std::vector<std::shared_ptr<dataset::Block>>;

inline std::vector<std::pair<float, uint32_t>>
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

inline std::tuple<std::vector<std::string>, std::vector<float>,
            std::vector<uint32_t>>
getPercentExplanationWithColumnNames(
    const std::vector<float>& gradients_ratio,
    std::vector<uint32_t> gradients_indices,
    std::unordered_map<uint32_t, std::string> num_to_name,const std::shared_ptr<dataset::GenericBatchProcessor>& generic_batch_processor) {
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
    auto [block,index_within_block] = generic_batch_processor->getBlockAndIndexWithinBlock(col.second);
    indices_within_block.push_back(index_within_block);
    auto column = block->getColumnNum();
    column_names.push_back(num_to_name[column]);
    gradient_percent_ratio.push_back((col.first / ratio_sum) * 100);
}
return std::make_tuple(column_names, gradient_percent_ratio,
                        indices_within_block);
}

}  // namespace thirdai::bolt