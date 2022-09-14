#pragma once
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <utility>

namespace thirdai::bolt {

using Blocks = std::vector<std::shared_ptr<dataset::Block>>;

// Here High significance means having high value of absolute ratio.

inline std::vector<std::pair<float, uint32_t>> sortGradientsBySignificance(
    std::vector<float> gradients_ratio,
    std::vector<uint32_t> gradients_indices) {
  std::vector<std::pair<float, uint32_t>> gradient_ratios_with_indices;
  for (uint32_t j = 0; j < gradients_ratio.size(); j++) {
    gradient_ratios_with_indices.push_back(
        std::make_pair(gradients_ratio[j], gradients_indices[j]));
  }
  auto func = [](std::pair<float, uint32_t> pair1,
                 std::pair<float, uint32_t> pair2) {
    return std::abs(pair1.first) > std::abs(pair2.first);
  };
  std::sort(gradient_ratios_with_indices.begin(),
            gradient_ratios_with_indices.end(), func);
  return gradient_ratios_with_indices;
}

/*
This function returns
1. Column names: list of column names corresponding to the responsible token.
2. Percentage Significance: list of values which tells us how much this token is
responsible.
3. Responsible token: The main thing in our RCA which gives us exact keyword is
responsible for this.

we get the column name and responsible token from generic batch processor itself
because that way, it will also be helpful tabular because it uses one block for
entire columns.
*/

inline std::tuple<std::vector<std::string>, std::vector<float>,
                  std::vector<std::string>>
getPercentExplanationWithColumnNames(
    const std::vector<float>& gradients_ratio,
    std::vector<uint32_t> gradients_indices,
    const std::unordered_map<uint32_t, std::string>& col_num_to_name,
    const std::shared_ptr<dataset::GenericBatchProcessor>&
        generic_batch_processor) {
  std::vector<std::pair<float, uint32_t>> gradients_ratio_with_indices =
      sortGradientsBySignificance(gradients_ratio,
                                  std::move(gradients_indices));
  float ratio_sum = 0;
  for (float gradient_ratio : gradients_ratio) {
    ratio_sum += std::abs(gradient_ratio);
  }
  std::vector<std::string> column_names;
  std::vector<float> gradient_significance;
  std::vector<std::string> words_responsible;
  for (const auto& col : gradients_ratio_with_indices) {
    auto [col_name, word_responsible] =
        generic_batch_processor->getResponsibleColumnAndInputKey(
            col.second, col_num_to_name);
    words_responsible.push_back(word_responsible);
    column_names.push_back(col_name);
    gradient_significance.push_back((col.first / ratio_sum) * 100);
  }
  return std::make_tuple(column_names, gradient_significance,
                         words_responsible);
}

}  // namespace thirdai::bolt