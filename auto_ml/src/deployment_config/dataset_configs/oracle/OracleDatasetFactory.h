#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include "Aliases.h"
#include "FeatureComposer.h"
#include "OracleConfig.h"
#include "TemporalContext.h"
#include "TemporalRelationshipsAutotuner.h"
#include <bolt/src/graph/nodes/Input.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/UserCountHistory.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::automl::deployment {

class OracleDatasetFactory final : public DatasetLoaderFactory {
 public:
  explicit OracleDatasetFactory(OracleConfigPtr config)
      : _config(std::move(config)),
        _temporal_relationships(TemporalRelationshipsAutotuner::autotune(
            _config->data_types, _config->provided_relationships,
            _config->lookahead)),
        _context(std::make_shared<TemporalContext>()) {
    ColumnNumberMap mock_column_number_map(_config->data_types);
    auto mock_processor = makeLabeledProcessor(mock_column_number_map);

    _input_dim = mock_processor->getInputDim();
    _label_dim = mock_processor->getLabelDim();
  }

  DatasetLoaderPtr getLabeledDatasetLoader(
      std::shared_ptr<dataset::DataLoader> data_loader, bool training) final {
    auto header = data_loader->nextLine();
    if (!header) {
      throw std::invalid_argument("The dataset must have a header.");
    }

    auto current_column_number_map =
        std::make_shared<ColumnNumberMap>(*header, /* delimiter= */ ',');
    if (!_column_number_map) {
      _column_number_map = std::move(current_column_number_map);
    } else if (!_column_number_map->equals(*current_column_number_map)) {
      throw std::invalid_argument("Column positions should not change.");
    }

    if (!_labeled_batch_processor) {
      _labeled_batch_processor = makeLabeledProcessor(*_column_number_map);
    }
    _context->initializeProcessor(_labeled_batch_processor);

    return std::make_unique<GenericDatasetLoader>(data_loader,
                                                  _labeled_batch_processor,
                                                  /* shuffle= */ training);
  }

  std::vector<BoltVector> featurizeInput(const std::string& input) final {
    initializeInferenceBatchProcessor();
    BoltVector vector;
    auto sample = dataset::ProcessorUtils::parseCsvRow(input, ',');
    _inference_batch_processor->makeInputVector(sample, vector);
    return {std::move(vector)};
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const std::vector<std::string>& inputs) final {
    initializeInferenceBatchProcessor();
    auto [input_batch, _] = _inference_batch_processor->createBatch(inputs);

    // We cannot use the initializer list because the copy constructor is
    // deleted for BoltBatch.
    std::vector<BoltBatch> batch_list;
    batch_list.emplace_back(std::move(input_batch));
    return batch_list;
  }

  std::vector<bolt::InputPtr> getInputNodes() final {
    return {bolt::Input::make(_input_dim)};
  }

  uint32_t getLabelDim() final { return _label_dim; }

  std::vector<std::string> listArtifactNames() const final {
    return {"temporal_context"};
  }

 protected:
  std::optional<Artifact> getArtifactImpl(const std::string& name) const final {
    if (name == "temporal_context") {
      return {_context};
    }
    return nullptr;
  }

 private:
  dataset::GenericBatchProcessorPtr makeLabeledProcessor(
      const ColumnNumberMap& column_number_map) {
    auto target_type = _config->data_types.at(_config->target);
    if (!target_type.isCategorical()) {
      throw std::invalid_argument(
          "Target column must be a categorical column.");
    }

    auto label_block = dataset::NumericalCategoricalBlock::make(
        /* col= */ column_number_map.at(_config->target),
        /* vocab_size= */ target_type.asCategorical().n_unique_classes);

    auto input_blocks =
        buildInputBlocks(/* column_numbers= */ column_number_map,
                         /* should_update_history= */ true);

    return dataset::GenericBatchProcessor::make(std::move(input_blocks),
                                                {label_block});
  }

  void initializeInferenceBatchProcessor() {
    if (_inference_batch_processor) {
      return;
    }
    if (!_column_number_map) {
      throw std::invalid_argument("Attempted inference before training.");
    }
    _inference_batch_processor = dataset::GenericBatchProcessor::make(
        buildInputBlocks(/* column_numbers= */ *_column_number_map,
                         /* should_update_history= */ false),
        /* label_blocks= */ {});
  }

  std::vector<dataset::BlockPtr> buildInputBlocks(
      const ColumnNumberMap& column_numbers, bool should_update_history) {
    std::vector<dataset::BlockPtr> blocks =
        FeatureComposer::makeSingleRowFeatureBlocks(
            *_config, _temporal_relationships, column_numbers, _vocabs);

    auto temporal_feature_blocks = FeatureComposer::makeTemporalFeatureBlocks(
        *_config, _temporal_relationships, column_numbers, _vocabs, *_context,
        should_update_history);

    blocks.insert(blocks.end(), temporal_feature_blocks.begin(),
                  temporal_feature_blocks.end());
    return blocks;
  }

  OracleConfigPtr _config;
  TemporalRelationships _temporal_relationships;
  TemporalContextPtr _context;
  std::unordered_map<std::string, dataset::ThreadSafeVocabularyPtr> _vocabs;
  ColumnNumberMapPtr _column_number_map;
  dataset::GenericBatchProcessorPtr _labeled_batch_processor;
  dataset::GenericBatchProcessorPtr _inference_batch_processor;
  uint32_t _input_dim;
  uint32_t _label_dim;

  // Private constructor for cereal.
  OracleDatasetFactory() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactory>(this), _config, _context,
            _vocabs, _column_number_map, _labeled_batch_processor,
            _inference_batch_processor, _input_dim, _label_dim);
  }
};

class OracleDatasetFactoryConfig final : public DatasetLoaderFactoryConfig {
 public:
  explicit OracleDatasetFactoryConfig(HyperParameterPtr<OracleConfigPtr> config)
      : _config(std::move(config)) {}

  DatasetLoaderFactoryPtr createDatasetState(
      const UserInputMap& user_specified_parameters) const final {
    auto config = _config->resolve(user_specified_parameters);

    return std::make_unique<OracleDatasetFactory>(config);
  }

 private:
  HyperParameterPtr<OracleConfigPtr> _config;

  // Private constructor for cereal.
  OracleDatasetFactoryConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactoryConfig>(this), _config);
  }
};

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::OracleDatasetFactoryConfig)

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::OracleDatasetFactory)