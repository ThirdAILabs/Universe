#pragma once

#include <bolt/src/graph/nodes/Input.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <exception>
#include <stdexcept>
#include <string_view>

namespace thirdai::automl::deployment_config {

class BasicClassificationDatasetState final : public DatasetState {
 public:
  BasicClassificationDatasetState(dataset::BlockPtr data_block,
                                  dataset::BlockPtr unlabeled_data_block,
                                  dataset::BlockPtr label_block, bool shuffle,
                                  char delimiter)
      : _labeled_batch_processor(
            std::make_shared<dataset::GenericBatchProcessor>(
                std::vector<dataset::BlockPtr>{std::move(data_block)},
                std::vector<dataset::BlockPtr>{std::move(label_block)},
                /* has_header= */ false, delimiter)),
        _unlabeled_batch_processor(
            std::make_shared<dataset::GenericBatchProcessor>(
                std::vector<dataset::BlockPtr>{std::move(unlabeled_data_block)},
                std::vector<dataset::BlockPtr>{},
                /* has_header= */ false, delimiter)),
        _shuffle(shuffle) {}

  DatasetLoaderPtr getDatasetLoader(
      std::shared_ptr<dataset::DataLoader> data_loader) final {
    return std::make_unique<GenericDatasetLoader>(
        data_loader, _labeled_batch_processor, _shuffle);
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

  std::vector<bolt::InputPtr> getInputNodes() final {
    return {bolt::Input::make(_unlabeled_batch_processor->getInputDim())};
  }

 private:
  dataset::GenericBatchProcessorPtr _labeled_batch_processor;
  dataset::GenericBatchProcessorPtr _unlabeled_batch_processor;
  bool _shuffle;
};

class BasicClassificationDatasetConfig final : public DatasetConfig {
 public:
  BasicClassificationDatasetConfig(BlockConfigPtr data_block,
                                   BlockConfigPtr label_block,
                                   HyperParameterPtr<bool> shuffle,
                                   HyperParameterPtr<std::string> delimiter)
      : _data_block(std::move(data_block)),
        _label_block(std::move(label_block)),
        _shuffle(std::move(shuffle)),
        _delimiter(std::move(delimiter)) {}

  DatasetStatePtr createDatasetState(
      const std::optional<std::string>& option,
      const UserInputMap& user_specified_parameters) const final {
    dataset::BlockPtr label_block = _label_block->getBlock(
        /* column= */ 0, option, user_specified_parameters);

    uint32_t data_start_col = label_block->expectedNumColumns();

    dataset::BlockPtr data_block = _data_block->getBlock(
        /* column= */ data_start_col, option, user_specified_parameters);

    dataset::BlockPtr unlabeled_data_block = _data_block->getBlock(
        /* column= */ 0, option, user_specified_parameters);

    bool shuffle = _shuffle->resolve(option, user_specified_parameters);
    std::string delimiter =
        _delimiter->resolve(option, user_specified_parameters);
    if (delimiter.size() != 1) {
      throw std::invalid_argument(
          "Expected delimiter to be a single character but recieved: '" +
          delimiter + "'.");
    }

    return std::make_unique<BasicClassificationDatasetState>(
        data_block, unlabeled_data_block, label_block, shuffle,
        delimiter.at(0));
  }

 private:
  BlockConfigPtr _data_block;
  BlockConfigPtr _label_block;
  HyperParameterPtr<bool> _shuffle;
  HyperParameterPtr<std::string> _delimiter;
};

}  // namespace thirdai::automl::deployment_config