#pragma once
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/callbacks/Callback.h>
#include <bolt_vector/src/BoltVector.h>
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
This function returns vector of 'PercentageResponsibleColumnAndInputKey' which
contains
1. percentage_significance : value which tells us how much this token is
responsible.
2. column_name : column name corresponding to the responsible token.
3. key_word responsible : The main thing in our RCA which gives us exact keyword
is responsible for this.

we get the column name and responsible token from generic batch processor itself
because that way, it will also be helpful for tabular because it uses one block
for entire columns.
*/
inline std::vector<dataset::PercentageResponsibleColumnAndInputKey>
getPercentExplanationWithColumnNames(
    const BoltGraphPtr& model, const BoltVector& input_vector,
    const std::vector<std::string_view>& columnar_sample,
    const std::shared_ptr<dataset::GenericBatchProcessor>&
        generic_batch_processor) {
  auto [gradients_indices, gradients_ratio] =
      model->getInputGradientSingle({input_vector});

  std::vector<std::pair<float, uint32_t>> gradients_ratio_with_indices =
      sortGradientsBySignificance(gradients_ratio,
                                  std::move(*gradients_indices));

  float ratio_sum = 0;
  for (float gradient_ratio : gradients_ratio) {
    ratio_sum += std::abs(gradient_ratio);
  }

  std::vector<dataset::PercentageResponsibleColumnAndInputKey>
      responsible_column_and_input_keys;

  for (const auto& col : gradients_ratio_with_indices) {
    dataset::ResponsibleInputs column_name_and_input_key =
        generic_batch_processor->explainIndex(col.second, columnar_sample);

    responsible_column_and_input_keys.push_back(
        {(col.first / ratio_sum) * 100, column_name_and_input_key});
  }

  return responsible_column_and_input_keys;
}

}  // namespace thirdai::bolt