#pragma once

#include <cereal/types/optional.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include "ConstructorUtilityTypes.h"
#include "SequentialClassifierConfig.h"
#include <dataset/src/DataLoader.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/blocks/UserCountHistory.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <dataset/src/utils/TimeUtils.h>
#include <sys/types.h>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace thirdai::bolt::sequential_classifier {

static inline void autotuneTemporalFeatures(
    SequentialClassifierConfig& config,
    std::map<std::string,
             std::vector<std::variant<std::string, TemporalConfig>>>&&
        temporal_relationships) {
  std::map<std::string, std::vector<TemporalConfig>> temporal_configs;
  for (const auto& [tracking_key, trackables] : temporal_relationships) {
    for (const auto& trackable : trackables) {
      if (std::holds_alternative<TemporalConfig>(trackable)) {
        temporal_configs[tracking_key].push_back(
            std::get<TemporalConfig>(trackable));
      } else {
        auto trackable_col = std::get<std::string>(trackable);
        if (config.data_types.at(trackable_col).isNumerical()) {
          uint32_t history_length =
              std::max(config.lookahead, static_cast<uint32_t>(1)) * 4;
          temporal_configs[tracking_key].push_back(
              TemporalConfig::numerical(trackable_col, history_length));
        } else if (config.data_types.at(trackable_col).isCategorical()) {
          temporal_configs[tracking_key].push_back(
              TemporalConfig::categorical(trackable_col, 1));
          temporal_configs[tracking_key].push_back(
              TemporalConfig::categorical(trackable_col, 2));
          temporal_configs[tracking_key].push_back(
              TemporalConfig::categorical(trackable_col, 5));
          temporal_configs[tracking_key].push_back(
              TemporalConfig::categorical(trackable_col, 10));
          temporal_configs[tracking_key].push_back(
              TemporalConfig::categorical(trackable_col, 25));
        } else {
          throw std::invalid_argument(
              trackable_col +
              " is neither numerical nor categorical. Only numerical and "
              "categorical columns can be tracked temporally.");
        }
      }
    }
  }
  config.temporal_tracking_relationships = temporal_configs;
}

static inline dataset::QuantityTrackingGranularity stringToGranularity(
    std::string&& granularity_string) {
  auto lower_granularity_string = utils::lower(granularity_string);
  if (lower_granularity_string == "daily" || lower_granularity_string == "d") {
    return dataset::QuantityTrackingGranularity::Daily;
  }
  if (lower_granularity_string == "weekly" || lower_granularity_string == "w") {
    return dataset::QuantityTrackingGranularity::Weekly;
  }
  if (lower_granularity_string == "biweekly" ||
      lower_granularity_string == "b") {
    return dataset::QuantityTrackingGranularity::Biweekly;
  }
  if (lower_granularity_string == "monthly" ||
      lower_granularity_string == "m") {
    return dataset::QuantityTrackingGranularity::Monthly;
  }
  throw std::invalid_argument(
      granularity_string +
      " is not a valid granularity option. The options are 'daily' / 'd', "
      "'weekly' / 'w', 'biweekly' / 'b', and 'monthly' / 'm',");
}

/**
 * Stores persistent states for data preprocessing.
 */
struct DataState {
  std::unordered_map<std::string, dataset::ThreadSafeVocabularyPtr>
      vocabs_by_column;

  /*
    We can technically have a vector instead of a map, but that makes
    the logic of buildPipeline a little harder to follow. Additionally,
    since the container contains pointers instead of the actual object,
    there is little benefit to having a vector instead.
  */
  std::unordered_map<std::string, dataset::ItemHistoryCollectionPtr>
      history_collections_by_id;

  std::unordered_map<std::string, dataset::QuantityHistoryTrackerPtr>
      quantity_histories_by_id;

  DataState() {}

  uint32_t n_index_single = 0;

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(vocabs_by_column, history_collections_by_id,
            quantity_histories_by_id, n_index_single);
  }
};

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

  ColumnNumberMap() {}

  explicit ColumnNumberMap(const std::map<std::string, DataType>& data_types) {
    uint32_t col_num = 0;
    for (const auto& [col_name, _] : data_types) {
      _name_to_num[col_name] = col_num;
      col_num++;
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

  size_t size() const { return _name_to_num.size(); }

  std::unordered_map<uint32_t, std::string> getColumnNumToColNameMap() {
    std::unordered_map<uint32_t, std::string> col_num_to_col_name;
    for (const auto& map : _name_to_num) {
      col_num_to_col_name[map.second] = map.first;
    }
    return col_num_to_col_name;
  }

 private:
  std::unordered_map<std::string, uint32_t> _name_to_num;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_name_to_num);
  }
};

class DataProcessing {
 public:
  static constexpr uint32_t BATCH_SIZE = 2048;

  static dataset::StreamingGenericDatasetLoader buildDataLoaderForFile(
      const SequentialClassifierConfig& config, DataState& state,
      const std::string& filename, char delimiter, bool for_training) {
    auto file_reader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, BATCH_SIZE);

    auto header = file_reader->nextLine();
    if (!header) {
      throw std::runtime_error("File header not found.");
    }

    ColumnNumberMap col_nums(*header, delimiter);

    auto input_blocks = buildInputBlocks(config, state, col_nums, for_training);

    std::vector<dataset::BlockPtr> label_blocks;
    if (!config.data_types.at(config.target).isCategorical()) {
      throw std::invalid_argument("Target must be a categorical column.");
    }
    auto target_meta = config.data_types.at(config.target).asCategorical();
    label_blocks.push_back(makeCategoricalBlock(
        config.target, target_meta.n_unique_classes, state, col_nums));

    return {file_reader,
            input_blocks,
            label_blocks,
            /* shuffle = */ for_training,
            /* config = */ {},
            /* has_header = */ false,  // since we already took the header.
            delimiter,
            /* parallel = */ false};  // We cannot properly capture sequences
                                      // in the dataset if we process it in
                                      // parallel.
  }

  static dataset::GenericBatchProcessorPtr buildSingleSampleBatchProcessor(
      const SequentialClassifierConfig& config, DataState& state,
      const ColumnNumberMap& col_nums, bool should_update_history) {
    auto input_blocks =
        buildInputBlocks(config, state, col_nums, /* for_training= */ false,
                         /* should_update_history= */ should_update_history);
    return dataset::GenericBatchProcessor::make(
        /* input_blocks= */ input_blocks, /* label_blocks= */ {});
  }

 private:
  static std::vector<dataset::BlockPtr> buildInputBlocks(
      const SequentialClassifierConfig& config, DataState& state,
      const ColumnNumberMap& col_nums, bool for_training,
      bool should_update_history = true) {
    auto [tracking_keys, trackables] = getTrackingKeysAndTrackableColumns(
        config.temporal_tracking_relationships);

    std::vector<dataset::BlockPtr> input_blocks;
    for (const auto& [col_name, data_type] : config.data_types) {
      if (data_type.isCategorical()) {
        auto categorical_meta = data_type.asCategorical();
        if (col_name != config.target &&
            (tracking_keys.count(col_name) || !trackables.count(col_name))) {
          input_blocks.push_back(makeCategoricalBlock(
              col_name, categorical_meta.n_unique_classes, state, col_nums));
        }
      }

      if (data_type.isNumerical()) {
        if (!trackables.count(col_name)) {
          input_blocks.push_back(dataset::DenseArrayBlock::make(
              /* start_col= */ col_nums.at(col_name), /* dim= */ 1));
        }
      }

      if (data_type.isText()) {
        auto text_meta = data_type.asText();
        if (text_meta.average_n_words && text_meta.average_n_words <= 15) {
          input_blocks.push_back(dataset::PairGramTextBlock::make(
              /* col= */ col_nums.at(col_name)));
        } else {
          input_blocks.push_back(dataset::UniGramTextBlock::make(
              /* col= */ col_nums.at(col_name)));
        }
      }

      if (data_type.isDate()) {
        input_blocks.push_back(
            dataset::DateBlock::make(/* col= */ col_nums.at(col_name)));
      }
    }

    if (config.temporal_tracking_relationships.empty()) {
      return input_blocks;
    }

    auto timestamp = getTimestamp(config);

    for (const auto& [tracking_key, trackables] :
         config.temporal_tracking_relationships) {
      uint32_t id = 0;
      for (const auto& trackable : trackables) {
        if (trackable.isCategorical()) {
          if (!config.data_types.at(trackable.columnName()).isCategorical()) {
            throw std::invalid_argument(
                "temporal.categorical can only be used with categorical "
                "columns.");
          }
          input_blocks.push_back(makeTemporalCategoricalBlock(
              /* id= */ tracking_key + "_" + std::to_string(id), config,
              tracking_key, trackable.asCategorical(), timestamp, state,
              col_nums, for_training, should_update_history));
        }

        if (trackable.isNumerical()) {
          if (!config.data_types.at(trackable.columnName()).isNumerical()) {
            throw std::invalid_argument(
                "temporal.numerical can only be used with numerical columns.");
          }
          input_blocks.push_back(makeTemporalNumericalBlock(
              /* id= */ tracking_key + "_" + std::to_string(id), config,
              tracking_key, trackable.asNumerical(), timestamp, state, col_nums,
              for_training, should_update_history));
        }

        id++;
      }
    }

    return input_blocks;
  }

  static std::pair<std::unordered_set<std::string>,
                   std::unordered_set<std::string>>
  getTrackingKeysAndTrackableColumns(
      const std::map<std::string, std::vector<TemporalConfig>>&
          temporal_tracking_relationships) {
    std::unordered_set<std::string> keys_set;
    std::unordered_set<std::string> trackables_set;
    for (const auto& [key, trackables] : temporal_tracking_relationships) {
      keys_set.insert(key);
      for (const auto& trackable : trackables) {
        trackables_set.insert(trackable.columnName());
      }
    }
    return {keys_set, trackables_set};
  }

  static std::string getTimestamp(const SequentialClassifierConfig& config) {
    std::optional<std::string> timestamp;
    if (!config.temporal_tracking_relationships.empty()) {
      for (const auto& [col_name, data_type] : config.data_types) {
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

  static dataset::BlockPtr makeCategoricalBlock(
      const std::string& col_name, uint32_t n_unique_classes, DataState& state,
      const ColumnNumberMap& col_nums) {
    auto& string_vocab = state.vocabs_by_column[col_name];

    if (!string_vocab) {
      string_vocab = dataset::ThreadSafeVocabulary::make(n_unique_classes);
    }
    return dataset::StringLookupCategoricalBlock::make(col_nums.at(col_name),
                                                       string_vocab);
  }

  static dataset::UserCountHistoryBlockPtr makeTemporalNumericalBlock(
      const std::string& id, const SequentialClassifierConfig& config,
      const std::string& tracking_key,
      const TemporalNumericalConfig& temporal_meta,
      const std::string& timestamp, DataState& state,
      const ColumnNumberMap& col_nums, bool for_training,
      bool should_update_history) {
    // Reset history if for training to prevent test data from leaking in.
    if (!state.quantity_histories_by_id[id] || for_training) {
      state.quantity_histories_by_id[id] =
          dataset::QuantityHistoryTracker::make(config.lookahead,
                                                temporal_meta.history_length,
                                                config.time_granularity);
    }

    return dataset::UserCountHistoryBlock::make(
        /* user_col= */ col_nums.at(tracking_key),
        /* count_col= */ col_nums.at(temporal_meta.column_name),
        /* timestamp_col= */ col_nums.at(timestamp),
        /* history= */ state.quantity_histories_by_id[id],
        /* should_update_history=*/should_update_history,
        /* include_current_row= */ temporal_meta.include_current_row);
  }

  // We pass in an ID because sequential blocks can corrupt each other's states.
  static dataset::BlockPtr makeTemporalCategoricalBlock(
      const std::string& id, const SequentialClassifierConfig& config,
      const std::string& tracking_key,
      const TemporalCategoricalConfig& temporal_meta,
      const std::string& timestamp, DataState& state,
      const ColumnNumberMap& col_nums, bool for_training,
      bool should_update_history) {
    uint32_t n_users =
        config.data_types.at(tracking_key).asCategorical().n_unique_classes;
    auto& user_vocab = state.vocabs_by_column[tracking_key];
    if (!user_vocab) {
      user_vocab = dataset::ThreadSafeVocabulary::make(n_users);
    }

    uint32_t n_items = config.data_types.at(temporal_meta.column_name)
                           .asCategorical()
                           .n_unique_classes;
    auto& item_vocab = state.vocabs_by_column[temporal_meta.column_name];
    if (!item_vocab) {
      item_vocab = dataset::ThreadSafeVocabulary::make(n_items);
    }

    // Reset history if for training to prevent test data from leaking in.
    if (!state.history_collections_by_id[id] || for_training) {
      state.history_collections_by_id[id] =
          dataset::ItemHistoryCollection::make(n_users);
    }

    int64_t time_lag = config.lookahead;
    time_lag *= dataset::QuantityHistoryTracker::granularityToSeconds(
        config.time_granularity);

    return dataset::UserItemHistoryBlock::make(
        /* user_col= */ col_nums.at(tracking_key),
        /* item_col= */ col_nums.at(temporal_meta.column_name),
        /* timestamp_col= */ col_nums.at(timestamp),
        /* user_id_map= */ user_vocab, /* item_id_map= */ item_vocab,
        /* records= */ state.history_collections_by_id[id],
        /* track_last_n= */ temporal_meta.track_last_n,
        /* should_update_history= */ should_update_history,
        /* include_current_row= */ temporal_meta.include_current_row,
        /* item_col_delimiter= */ std::nullopt,
        /* time_lag= */ time_lag);
  }
};

}  // namespace thirdai::bolt::sequential_classifier