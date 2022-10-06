#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include "OracleConfig.h"
#include "TemporalContext.h"
#include <bolt/src/auto_classifiers/sequential_classifier/ConstructorUtilityTypes.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt_vector/src/BoltVector.h>
#include <_types/_uint64_t.h>
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

using DataType = bolt::sequential_classifier::DataType;

class ColumnNumberMap {
 public:
  ColumnNumberMap(const std::string& header, char delimiter) {
    auto header_columns =
        dataset::ProcessorUtils::parseCsvRow(header, delimiter);
    for (uint32_t col_num = 0; col_num < header_columns.size(); col_num++) {
      std::string col_name(header_columns[col_num]);
      _name_to_num[col_name] = col_num;
    }
  }

  uint32_t at(const std::string& col_name) const {
    if (_name_to_num.count(col_name) == 0) {
      std::stringstream error_ss;
      error_ss << "Expected a column named '" << col_name
               << "' in header but could not find it.";
      throw std::runtime_error(error_ss.str());
    }
    return _name_to_num.at(col_name);
  }

  bool equals(const ColumnNumberMap& other) {
    return other._name_to_num == _name_to_num;
  }

  size_t size() const { return _name_to_num.size(); }

 private:
  std::unordered_map<std::string, uint32_t> _name_to_num;

  // Private constructor for cereal
  ColumnNumberMap() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_name_to_num);
  }
};

using ColumnNumberMapPtr = std::shared_ptr<ColumnNumberMap>;

class VocabularyManager {
 public:
  dataset::ThreadSafeVocabularyPtr forColumn(const std::string& column_name,
                                             uint32_t vocab_size) {
    if (!_col_name_to_vocab.count(column_name)) {
      _col_name_to_vocab[column_name] =
          dataset::ThreadSafeVocabulary::make(vocab_size);
    }
    return _col_name_to_vocab.at(column_name);
  }

 private:
  std::unordered_map<std::string, dataset::ThreadSafeVocabularyPtr>
      _col_name_to_vocab;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_col_name_to_vocab);
  }
};

class OracleDatasetFactory final : public DatasetLoaderFactory {
 public:
  OracleDatasetFactory(OracleConfigPtr config, TemporalContextPtr context)
      : _config(std::move(config)), _context(std::move(context)) {
    if (!_config->temporal_tracking_relationships.empty() &&
        _context->isNone()) {
      throw std::invalid_argument(
          "Oracle requires a temporal context object if temporal tracking "
          "relationships are specified.");
    }
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

    initializeLabeledBatchProcessor();

    return std::make_unique<GenericDatasetLoader>(data_loader,
                                                  _labeled_batch_processor,
                                                  /* shuffle= */ training);
  }

  std::vector<BoltVector> featurizeInput(const std::string& input) final {
    // Build an inference batch processor if it does not exist yet
    // Think about serialization.
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
    std::string header;
    for (const auto& [col_name, _] : _config->data_types) {
      header += col_name + ",";
    }
    ColumnNumberMap mock_column_number_map(header, /* delimiter= */ ',');
    auto input_dim =
        dataset::GenericBatchProcessor::make(
            buildInputBlocks(/* column_numbers= */ mock_column_number_map,
                             /* should_update_history= */ false),
            /* label_blocks= */ {})
            ->getInputDim();
    return {bolt::Input::make(input_dim)};
  }

 private:
  void initializeLabeledBatchProcessor() {
    if (_labeled_batch_processor || !_column_number_map) {
      return;
    }

    auto input_blocks =
        buildInputBlocks(/* column_numbers= */ *_column_number_map,
                         /* should_update_history= */ true);

    auto target_type = _config->data_types.at(_config->target);
    if (!target_type.isCategorical()) {
      throw std::invalid_argument(
          "Target column must be a categorical column.");
    }

    auto label_block =
        makeCategoricalBlock(/* column_numbers= */ *_column_number_map,
                             /* column_name= */ _config->target);

    _labeled_batch_processor = dataset::GenericBatchProcessor::make(
        std::move(input_blocks), {label_block});

    _context->initializeProcessor(_labeled_batch_processor);
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
        input_blocks.push_back(makeTextBlock(column_numbers, col_name));
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

    // Order of tracking keys is always consistent because
    // temporal_tracking_relationships is an ordered map.
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
              id, column_numbers, tracking_key, config.columnName(), timestamp,
              config, should_update_history));
        }

        if (config.isNumerical()) {
          input_blocks.push_back(makeTemporalNumericalBlock(
              id, column_numbers, tracking_key, config.columnName(), timestamp,
              config, should_update_history));
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
                                         const std::string& column_name) {
    auto vocab_size =
        _config->data_types.at(column_name).asCategorical().n_unique_classes;

    return dataset::StringLookupCategoricalBlock::make(
        /* col= */ column_numbers.at(column_name),
        /* vocab= */ _vocabs.forColumn(column_name, vocab_size));
  }

  dataset::BlockPtr makeTextBlock(const ColumnNumberMap& column_numbers,
                                  const std::string& column_name) {
    auto text_meta = _config->data_types.at(column_name).asText();
    if (text_meta.average_n_words && text_meta.average_n_words <= 15) {
      return dataset::PairGramTextBlock::make(
          /* col= */ column_numbers.at(column_name));
    }
    return dataset::UniGramTextBlock::make(
        /* col= */ column_numbers.at(column_name));
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
      const std::string& key_column, const std::string& tracked_column,
      const std::string& timestamp_column,
      const TemporalConfig& temporal_config, bool should_update_history) {
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
        /* user_id_map= */ _vocabs.forColumn(key_column, key_vocab_size),
        /* item_id_map= */
        _vocabs.forColumn(tracked_column, tracked_vocab_size),
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
      const std::string& key_column, const std::string& tracked_column,
      const std::string& timestamp_column,
      const TemporalConfig& temporal_config, bool should_update_history) {
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

  OracleConfigPtr _config;
  TemporalContextPtr _context;
  VocabularyManager _vocabs;
  ColumnNumberMapPtr _column_number_map;
  dataset::GenericBatchProcessorPtr _labeled_batch_processor;
  dataset::GenericBatchProcessorPtr _inference_batch_processor;

  // Private constructor for cereal.
  OracleDatasetFactory() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactory>(this), _config, _context,
            _vocabs, _column_number_map, _labeled_batch_processor,
            _inference_batch_processor);
  }
};

class OracleDatasetFactoryConfig final : public DatasetLoaderFactoryConfig {
 public:
  OracleDatasetFactoryConfig(HyperParameterPtr<OracleConfigPtr> config,
                             HyperParameterPtr<TemporalContextPtr> context)
      : _config(std::move(config)), _context(std::move(context)) {}

  DatasetLoaderFactoryPtr createDatasetState(
      const UserInputMap& user_specified_parameters) const final {
    auto config = _config->resolve(user_specified_parameters);
    auto context = _context->resolve(user_specified_parameters);

    return std::make_unique<OracleDatasetFactory>(config, context);
  }

 private:
  HyperParameterPtr<OracleConfigPtr> _config;
  HyperParameterPtr<TemporalContextPtr> _context;

  // Private constructor for cereal.
  OracleDatasetFactoryConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactoryConfig>(this), _config,
            _context);
  }
};

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::OracleDatasetFactoryConfig)

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::OracleDatasetFactory)