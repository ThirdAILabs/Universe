#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt {

using Blocks = std::vector<std::shared_ptr<dataset::Block>>;

/**
 * @brief Given the gradients ratios and indices, sort the gradients retaining
 * the indices.
 *
 * @return Returns vector of pairs(gradients ratios and indices) sorted in the
 * descending order of absolute values.
 */
inline std::vector<std::pair<float, uint32_t>> sortGradientsBySignificance(
    std::vector<float> gradients_ratio,
    std::optional<std::vector<uint32_t>> gradients_indices) {
  std::vector<uint32_t> indices(gradients_ratio.size());
  if (!gradients_indices) {
    std::iota(indices.begin(), indices.end(), 0);
  } else {
    indices = *gradients_indices;
  }
  std::vector<std::pair<float, uint32_t>> gradient_ratios_with_indices;
  for (uint32_t j = 0; j < gradients_ratio.size(); j++) {
    gradient_ratios_with_indices.push_back(
        std::make_pair(gradients_ratio[j], indices[j]));
  }
  auto func = [](std::pair<float, uint32_t> pair1,
                 std::pair<float, uint32_t> pair2) {
    return std::abs(pair1.first) > std::abs(pair2.first);
  };
  std::sort(gradient_ratios_with_indices.begin(),
            gradient_ratios_with_indices.end(), func);
  return gradient_ratios_with_indices;
}

/**
 * @brief Get the gradients information from the model with respect to given
 * input vector and sort the gradients ratios with maintaining the indices and
 * for each gradient value with index pair, get corresponding column number and
 * key word from the featurizer given.
 *
 * @param gradient_indices indices of the input vector that will be explained.
 * Note that this is an optional that only has a value if the vector is sparse.
 *
 * @param gradients_ratio Gradients normalized by input values. e.g. if input
 * values = [1.0, 2.0, 3.0, 4.0] and gradients = [0.2, 0.2, 0.6, 0.1], then
 * gradients_ratio = [0.2, 0.1, 0.2, 0.025]
 *
 * @param input_row The string view of input which can be used for getting the
 * exact key words responsible from blocks when user calls explain method,
 * rather than overloading buildsegment method which might affect the
 * threadsafety.
 *
 * @param generic_featurizer The featurizer from which we can get
 * column number and keyword responsible for the given index.
 *
 * @return vector of Explanation structs, sorted in descending
 * order of their significance percentages.
 */
inline std::vector<dataset::Explanation> getSignificanceSortedExplanations(
    const std::optional<std::vector<uint32_t>>& gradients_indices,
    const std::vector<float>& gradients_ratio,
    dataset::ColumnarInputSample& input,
    const std::shared_ptr<dataset::TabularFeaturizer>& generic_featurizer) {
  std::vector<std::pair<float, uint32_t>> gradients_ratio_with_indices =
      sortGradientsBySignificance(gradients_ratio, gradients_indices);

  float ratio_sum = 0;
  for (float gradient_ratio : gradients_ratio) {
    ratio_sum += std::abs(gradient_ratio);
  }

  if (ratio_sum == 0) {
    throw std::invalid_argument(
        "The model has not learned enough to give explanations. Try "
        "decreasing the learning rate.");
  }

  std::vector<dataset::Explanation> explanations;

  // We rebuild the vector to get the index to segment feature map.
  // TODO(Geordie): Reuse information from the forward pass.
  auto index_to_segment_feature =
      generic_featurizer->getIndexToSegmentFeatureMap(input);

  for (const auto& [ratio, index] : gradients_ratio_with_indices) {
    if (ratio) {
      dataset::Explanation explanation_for_index =
          generic_featurizer->explainFeature(
              input,
              /* segment_feature= */ index_to_segment_feature.at(index));
      explanation_for_index.percentage_significance = (ratio / ratio_sum) * 100;
      explanations.push_back(explanation_for_index);
    }
  }

  return explanations;
}

}  // namespace thirdai::bolt