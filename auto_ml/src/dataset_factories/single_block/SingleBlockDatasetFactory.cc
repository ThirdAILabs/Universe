#include "SingleBlockDatasetFactory.h"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/InputTypes.h>

namespace thirdai::automl::data {

std::vector<BoltVector> SingleBlockDatasetFactory::featurizeInput(
    const std::string& input) {
  std::vector<std::string_view> input_vector = {
      std::string_view(input.data(), input.length())};
  dataset::RowSampleRef input_vector_ref(input_vector);
  return {_unlabeled_featurizer->makeInputVector(input_vector_ref)};
}

std::vector<BoltBatch> SingleBlockDatasetFactory::featurizeInputBatch(
    const std::vector<std::string>& inputs) {
  auto batches = _unlabeled_featurizer->featurize(inputs);

  // We cannot use the initializer list because the copy constructor is
  // deleted for BoltBatch.
  std::vector<BoltBatch> batch_list;
  batch_list.emplace_back(std::move(batches.at(0)));
  return batch_list;
}

uint32_t SingleBlockDatasetFactory::labelToNeuronId(
    std::variant<uint32_t, std::string> label) {
  if (std::holds_alternative<uint32_t>(label)) {
    return std::get<uint32_t>(label);
  }

  throw std::invalid_argument(
      "This model does not support string labels; label must be a "
      "non-negative integer.");
}

std::vector<dataset::Explanation> SingleBlockDatasetFactory::explain(
    const std::optional<std::vector<uint32_t>>& gradients_indices,
    const std::vector<float>& gradients_ratio, const std::string& sample) {
  dataset::CsvSampleRef sample_ref(sample, _delimiter);
  return bolt::getSignificanceSortedExplanations(
      gradients_indices, gradients_ratio, sample_ref, _unlabeled_featurizer);
}

}  // namespace thirdai::automl::data

CEREAL_REGISTER_TYPE(thirdai::automl::data::SingleBlockDatasetFactory)
