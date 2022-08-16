#pragma once
#include <dataset/src/blocks/BlockInterface.h>

#include <utility>


namespace thirdai::bolt {

using Blocks = std::vector<std::shared_ptr<dataset::Block>>;

class RootCauseAnalysis {
    public:
    explicit RootCauseAnalysis(Blocks input_blocks) : _input_blocks(std::move(input_blocks)) {
        std::vector<uint32_t> offsets;
        offsets.push_back(0); 
        for(const auto& block: _input_blocks) {
            offsets.push_back(block->featureDim());
        }
        _offsets = std::move(offsets);
    }

    uint32_t getIndexOfBlock(uint32_t index) {
        auto iter = std::upper_bound(_offsets.begin(), _offsets.end(), index);
        return (iter - _offsets.begin() - 1);
    }

    static std::vector<std::pair<float, uint32_t>> makeGradientRatiosWithIndices(std::vector<float> gradients_ratio, std::vector<uint32_t> gradients_indices) {
        std::vector<std::pair<float, uint32_t>> gradient_ratios_with_indices;
        // float sum = 0;
        for (uint32_t j = 0; j < gradients_ratio.size(); j++) {
            // sum += abs(gradients_ratio[j]);
            gradient_ratios_with_indices.push_back(
                std::make_pair(gradients_ratio[j], gradients_indices[j]));
        }
        return gradient_ratios_with_indices;
        // ratio_sums.push_back(sum);
    }

    std::vector<std::vector<std::pair<float, uint32_t>>> getGradientRatiosSorted() {}


    private:
    Blocks _input_blocks;
    std::vector<uint32_t> _offsets;

};

} // namespace thirdai::bolt