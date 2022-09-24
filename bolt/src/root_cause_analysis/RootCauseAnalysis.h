#pragma once
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/callbacks/Callback.h>
#include <bolt_vector/src/BoltVector.h>
#include <_types/_uint32_t.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
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
 * key word from the batch processor given.
 *
 * @param model The model to use for RCA.
 *
 * @param input_row The string view of input which can be used for getting the
 * exact key words responsible from blocks when user calls explain method,
 * rather than overloading buildsegment method which might affect the
 * threadsafety.
 *
 * @param generic_batch_processor The batchprocessor from which we can get
 * column number and keyword responsible for the given index.
 *
 * @return vector of Explanation structs, sorted in descending
 * order of their significance percentages.
 */
inline std::vector<dataset::Explanation> getSignificanceSortedExplanations(
    const BoltGraphPtr& model, const BoltVector& input_vector,
    const std::vector<std::string_view>& input_row,
    const std::shared_ptr<dataset::GenericBatchProcessor>&
        generic_batch_processor,
    std::optional<uint32_t> neuron_to_explain = std::nullopt) {
  auto [gradients_indices, gradients_ratio] =
      model->getInputGradientSingle({input_vector}, true, neuron_to_explain);

  std::vector<std::pair<float, uint32_t>> gradients_ratio_with_indices =
      sortGradientsBySignificance(gradients_ratio, gradients_indices);

  float ratio_sum = 0;
  for (float gradient_ratio : gradients_ratio) {
    ratio_sum += std::abs(gradient_ratio);
  }

  std::vector<dataset::Explanation> explanations;

  for (const auto& [ratio, index] : gradients_ratio_with_indices) {
    dataset::Explanation explanation_for_index =
        generic_batch_processor->explainIndex(index, input_row);
    explanation_for_index.percentage_significance = (ratio / ratio_sum) * 100;
    explanations.push_back(explanation_for_index);
  }

  return explanations;
}

}  // namespace thirdai::bolt