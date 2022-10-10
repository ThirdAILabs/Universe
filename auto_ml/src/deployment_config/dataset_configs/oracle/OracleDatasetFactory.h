#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include "OracleConfig.h"
#include "TemporalContext.h"
#include <bolt/src/auto_classifiers/sequential_classifier/ConstructorUtilityTypes.h>
#include <bolt/src/auto_classifiers/sequential_classifier/SequentialUtils.h>
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

using ColumnNumberMap = bolt::sequential_classifier::ColumnNumberMap;
using ColumnNumberMapPtr = std::shared_ptr<ColumnNumberMap>;

class OracleDatasetFactory final : public DatasetLoaderFactory {
 public:
  explicit OracleDatasetFactory(OracleConfigPtr config)
      : _config(std::move(config)),
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

    auto label_block = makeCategoricalBlock(
        /* column_numbers= */ column_number_map,
        /* column_name= */ _config->target, /* from_string= */ false);

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
    std::vector<dataset::BlockPtr> input_blocks;
    addStaticInputBlocks(column_numbers, input_blocks);
    addTemporalInputBlocks(column_numbers, input_blocks, should_update_history);
    return input_blocks;
  }

  std::vector<dataset::BlockPtr> addStaticInputBlocks(
      const ColumnNumberMap& column_numbers,
      std::vector<dataset::BlockPtr>& input_blocks) {
    auto trackable_oolumns = trackableColumns();

    // Order of column names and data types is always consistent because
    // data_types is an ordered map.
    for (const auto& [col_name, data_type] : _config->data_types) {
      if (data_type.isCategorical()) {
        if (col_name != _config->target &&
            (isTrackableKay(col_name) || !trackable_oolumns.count(col_name))) {
          input_blocks.push_back(
              makeCategoricalBlock(column_numbers, col_name));
        }
      }

      if (data_type.isNumerical()) {
        if (!trackable_oolumns.count(col_name)) {
          input_blocks.push_back(dataset::DenseArrayBlock::make(
              /* start_col= */ column_numbers.at(col_name), /* dim= */ 1));
        }
      }

      if (data_type.isText()) {
        auto text_meta = _config->data_types.at(col_name).asText();
        auto col_num = column_numbers.at(col_name);
        if (text_meta.average_n_words && text_meta.average_n_words <= 15) {
          input_blocks.push_back(dataset::PairGramTextBlock::make(col_num));
        } else {
          input_blocks.push_back(dataset::UniGramTextBlock::make(col_num));
        }
      }

      if (data_type.isDate()) {
        input_blocks.push_back(dataset::DateBlock::make(
            /* col= */ column_numbers.at(col_name)));
      }
    }
    return input_blocks;
  }

  void addTemporalInputBlocks(const ColumnNumberMap& column_numbers,
                              std::vector<dataset::BlockPtr>& input_blocks,
                              bool should_update_history) {
    auto timestamp = getTimestamp();

    /*
      Order of tracking keys is always consistent because
      temporal_tracking_relationships is an ordered map.
      Therefore, the order of ids is also consistent.
    */
    uint32_t id = 0;
    for (const auto& [tracking_key, temporal_configs] :
         _config->temporal_tracking_relationships) {
      auto tracking_key_type = _config->data_types.at(tracking_key);
      if (!tracking_key_type.isCategorical()) {
        throw std::invalid_argument("Tracking keys must be categorical.");
      }

      for (const auto& config : temporal_configs) {
        if (config.isCategorical()) {
          input_blocks.push_back(makeTemporalCategoricalBlock(
              id, column_numbers, tracking_key, timestamp, config,
              should_update_history));
        }

        if (config.isNumerical()) {
          input_blocks.push_back(makeTemporalNumericalBlock(
              id, column_numbers, tracking_key, timestamp, config,
              should_update_history));
        }

        id++;
      }
    }
  }

  std::unordered_set<std::string> trackableColumns() {
    std::unordered_set<std::string> trackables_set;
    for (const auto& [_, trackables] :
         _config->temporal_tracking_relationships) {
      for (const auto& trackable : trackables) {
        trackables_set.insert(trackable.columnName());
      }
    }
    return trackables_set;
  }

  bool isTrackableKay(const std::string& column_name) {
    return _config->temporal_tracking_relationships.count(column_name);
  }

  dataset::BlockPtr makeCategoricalBlock(const ColumnNumberMap& column_numbers,
                                         const std::string& column_name,
                                         bool from_string = true) {
    auto vocab_size =
        _config->data_types.at(column_name).asCategorical().n_unique_classes;

    uint32_t col = column_numbers.at(column_name);

    if (from_string) {
      return dataset::StringLookupCategoricalBlock::make(
          /* col= */ col,
          /* vocab= */ vocabForColumn(column_name, vocab_size));
    }

    return dataset::NumericalCategoricalBlock::make(
        /* col= */ col, /* n_classes= */ vocab_size);
  }

  std::string getTimestamp() {
    std::optional<std::string> timestamp;
    if (!_config->temporal_tracking_relationships.empty()) {
      for (const auto& [col_name, data_type] : _config->data_types) {
        if (data_type.isDate()) {
          if (timestamp) {
            throw std::invalid_argument(
                "There can only be one timestamp column.");
          }
          timestamp = col_name;
        }
      }
    }
    if (!timestamp) {
      throw std::invalid_argument(
          "There has to be a timestamp column in order to use temporal "
          "tracking relationships.");
    }
    return *timestamp;
  }

  dataset::BlockPtr makeTemporalCategoricalBlock(
      uint32_t id, const ColumnNumberMap& column_numbers,
      const std::string& key_column, const std::string& timestamp_column,
      const OracleTemporalConfig& temporal_config, bool should_update_history) {
    const auto& tracked_column = temporal_config.columnName();

    if (!_config->data_types.at(tracked_column).isCategorical()) {
      throw std::invalid_argument(
          "temporal.categorical can only be used with categorical "
          "columns.");
    }

    auto key_vocab_size =
        _config->data_types.at(key_column).asCategorical().n_unique_classes;
    auto tracked_vocab_size =
        _config->data_types.at(tracked_column).asCategorical().n_unique_classes;
    auto temporal_meta = temporal_config.asCategorical();

    int64_t time_lag = _config->lookahead;
    time_lag *= dataset::QuantityHistoryTracker::granularityToSeconds(
        _config->time_granularity);

    return dataset::UserItemHistoryBlock::make(
        /* user_col= */ column_numbers.at(key_column),
        /* item_col= */ column_numbers.at(tracked_column),
        /* timestamp_col= */ column_numbers.at(timestamp_column),
        /* user_id_map= */ vocabForColumn(key_column, key_vocab_size),
        /* item_id_map= */
        vocabForColumn(tracked_column, tracked_vocab_size),
        /* records= */
        _context->categoricalHistoryForId(id, /* n_users= */ key_vocab_size),
        /* track_last_n= */ temporal_meta.track_last_n,
        /* should_update_history= */ should_update_history,
        /* inlcude_current_row= */ temporal_meta.include_current_row,
        /* item_col_delimiter= */ std::nullopt,
        /* time_lag= */ time_lag);
  }

  dataset::BlockPtr makeTemporalNumericalBlock(
      uint32_t id, const ColumnNumberMap& column_numbers,
      const std::string& key_column, const std::string& timestamp_column,
      const OracleTemporalConfig& temporal_config, bool should_update_history) {
    const auto& tracked_column = temporal_config.columnName();

    if (!_config->data_types.at(tracked_column).isNumerical()) {
      throw std::invalid_argument(
          "temporal.numerical can only be used with numerical columns.");
    }

    auto temporal_meta = temporal_config.asNumerical();

    auto numerical_history = _context->numericalHistoryForId(
        /* id= */ id,
        /* lookahead= */ _config->lookahead,
        /* history_length= */ temporal_meta.history_length,
        /* time_granularity= */ _config->time_granularity);

    return dataset::UserCountHistoryBlock::make(
        /* user_col= */ column_numbers.at(key_column),
        /* count_col= */ column_numbers.at(tracked_column),
        /* timestamp_col= */ column_numbers.at(timestamp_column),
        /* history= */ numerical_history,
        /* should_update_history= */ should_update_history,
        /* include_current_row= */ temporal_meta.include_current_row);
  }

  dataset::ThreadSafeVocabularyPtr& vocabForColumn(
      const std::string& column_name, uint32_t vocab_size) {
    if (!_vocabs.count(column_name)) {
      _vocabs[column_name] = dataset::ThreadSafeVocabulary::make(vocab_size);
    }
    return _vocabs.at(column_name);
  }

  OracleConfigPtr _config;
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