#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include "DatasetFactory.h"
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <utils/StringManipulation.h>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace thirdai::automl::data {

class SingleBlockDatasetFactory final : public DatasetLoaderFactory {
 public:
  SingleBlockDatasetFactory(dataset::BlockPtr data_block,
                            dataset::BlockPtr unlabeled_data_block,
                            dataset::BlockPtr label_block, bool shuffle,
                            char delimiter, bool has_header)
      : _labeled_batch_processor(
            std::make_shared<dataset::GenericBatchProcessor>(
                std::vector<dataset::BlockPtr>{std::move(data_block)},
                std::vector<dataset::BlockPtr>{std::move(label_block)},
                /* has_header= */ has_header, delimiter)),
        _unlabeled_batch_processor(
            std::make_shared<dataset::GenericBatchProcessor>(
                std::vector<dataset::BlockPtr>{std::move(unlabeled_data_block)},
                std::vector<dataset::BlockPtr>{},
                /* has_header= */ has_header, delimiter)),
        _shuffle(shuffle),
        _delimiter(delimiter) {}

  DatasetLoaderPtr getLabeledDatasetLoader(
      std::shared_ptr<dataset::DataLoader> data_loader, bool training) final {
    return std::make_unique<GenericDatasetLoader>(
        data_loader, _labeled_batch_processor, _shuffle && training);
  }

  std::vector<BoltVector> featurizeInput(const std::string& input) final {
    BoltVector output;

    std::vector<std::string_view> input_vector = {
        std::string_view(input.data(), input.length())};
    if (auto exception =
            _unlabeled_batch_processor->makeInputVector(input_vector, output)) {
      std::rethrow_exception(exception);
    }
    return {std::move(output)};
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const std::vector<std::string>& inputs) final {
    auto [batch, _] = _unlabeled_batch_processor->createBatch(inputs);

    // We cannot use the initializer list because the copy constructor is
    // deleted for BoltBatch.
    std::vector<BoltBatch> batch_list;
    batch_list.emplace_back(std::move(batch));
    return batch_list;
  }

  uint32_t labelToNeuronId(std::variant<uint32_t, std::string> label) final {
    if (std::holds_alternative<uint32_t>(label)) {
      return std::get<uint32_t>(label);
    }

    throw std::invalid_argument(
        "This model does not support string labels; label must be a "
        "non-negative integer.");
  }

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio,
      const std::string& sample) final {
    auto input_row = dataset::ProcessorUtils::parseCsvRow(sample, _delimiter);
    return bolt::getSignificanceSortedExplanations(gradients_indices,
                                                   gradients_ratio, input_row,
                                                   _unlabeled_batch_processor);
  }

  std::vector<bolt::InputPtr> getInputNodes() final {
    return {bolt::Input::make(_unlabeled_batch_processor->getInputDim())};
  }

  uint32_t getLabelDim() final {
    return _labeled_batch_processor->getLabelDim();
  }

 private:
  dataset::GenericBatchProcessorPtr _labeled_batch_processor;
  dataset::GenericBatchProcessorPtr _unlabeled_batch_processor;
  bool _shuffle;
  char _delimiter;

  // Private constructor for cereal.
  SingleBlockDatasetFactory() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactory>(this),
            _labeled_batch_processor, _unlabeled_batch_processor, _shuffle,
            _delimiter);
  }
};

}  // namespace thirdai::automl::data

CEREAL_REGISTER_TYPE(thirdai::automl::data::SingleBlockDatasetFactory)
