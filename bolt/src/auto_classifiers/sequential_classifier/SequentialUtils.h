#pragma once

#include <cereal/types/optional.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <dataset/src/DataLoader.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/blocks/Trend.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/utils/CountHistoryIndex.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <sys/types.h>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::bolt::sequential_classifier {

// A pair of (column name, num unique classes)
using CategoricalPair = std::pair<std::string, uint32_t>;
// A tuple of (column name, num unique classes, track last N)
using SequentialTriplet = std::tuple<std::string, uint32_t, uint32_t>;
// A tuple of (column name, num unique classes, track last N)
using DenseSequentialQuadruplet = std::tuple<std::string, /* lookahead= */ uint32_t, /* lookback= */ uint32_t, /* period= */ uint32_t>;

/**
 * Stores the dataset configuration.
 */
struct Schema {
  CategoricalPair user;
  CategoricalPair target;
  std::string timestamp_col_name;
  std::vector<std::string> static_text_col_names;
  std::vector<CategoricalPair> static_categorical;
  std::vector<SequentialTriplet> sequential;
  std::vector<DenseSequentialQuadruplet> dense_sequential;
  std::optional<char> multi_class_delim;

  Schema() {}

  std::unordered_set<std::string> allColumnNames() const {
    std::unordered_set<std::string> col_names;
    col_names.insert(std::get<0>(user));
    col_names.insert(std::get<0>(target));
    col_names.insert(timestamp_col_name);
    for (const auto& text_col_name : static_text_col_names) {
      col_names.insert(text_col_name);
    }
    for (const auto& [cat_col_name, _1] : static_categorical) {
      col_names.insert(cat_col_name);
    }
    for (const auto& [seq_col_name, _1, _2] : sequential) {
      col_names.insert(seq_col_name);
    }
    return col_names;
  }

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(user, target, timestamp_col_name, static_text_col_names,
            static_categorical, sequential, multi_class_delim);
  }
};

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
  std::unordered_map<uint32_t, dataset::ItemHistoryCollectionPtr>
      history_collections_by_id;

  std::unordered_map<uint32_t, dataset::CountHistoryIndexPtr> count_histories_by_id;

  DataState() {}

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(vocabs_by_column, history_collections_by_id);
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

  explicit ColumnNumberMap(const Schema& schema) {
    auto columns = schema.allColumnNames();
    uint32_t col_num = 0;
    for (const auto& col_name : columns) {
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
};

class Pipeline {
 public:
  static constexpr uint32_t BATCH_SIZE = 2048;

  static dataset::StreamingGenericDatasetLoader buildForFile(
      const Schema& schema, DataState& state, const std::string& filename,
      char delimiter, bool for_training) {
    auto file_reader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, BATCH_SIZE);

    auto header = file_reader->nextLine();
    if (!header) {
      throw std::runtime_error("File header not found.");
    }

    ColumnNumberMap col_nums(*header, delimiter);

    auto input_blocks = buildInputBlocks(schema, state, col_nums, for_training);

    std::vector<dataset::BlockPtr> label_blocks;
    label_blocks.push_back(makeCategoricalBlock(schema.target, state, col_nums,
                                                schema.multi_class_delim));

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

  static dataset::GenericBatchProcessorPtr buildSingleInferenceBatchProcessor(
      const Schema& schema, DataState& state, const ColumnNumberMap& col_nums) {
    auto input_blocks = buildInputBlocks(schema, state, col_nums, false);
    return dataset::GenericBatchProcessor::make(
        /* input_blocks= */ input_blocks, /* label_blocks= */ {});
  }

 private:
  static std::vector<dataset::BlockPtr> buildInputBlocks(
      const Schema& schema, DataState& state, const ColumnNumberMap& col_nums,
      bool for_training) {
    std::vector<dataset::BlockPtr> input_blocks;
    input_blocks.push_back(makeCategoricalBlock(schema.user, state, col_nums,
                                                /* delimiter= */ std::nullopt));

    input_blocks.push_back(std::make_shared<dataset::DateBlock>(
        col_nums.at(schema.timestamp_col_name)));

    for (const auto& text_col_name : schema.static_text_col_names) {
      input_blocks.push_back(
          dataset::UniGramTextBlock::make(col_nums.at(text_col_name)));
    }

    for (const auto& categorical : schema.static_categorical) {
      input_blocks.push_back(makeCategoricalBlock(categorical, state, col_nums,
                                                  schema.multi_class_delim));
    }

    for (uint32_t seq_idx = 0; seq_idx < schema.sequential.size(); seq_idx++) {
      input_blocks.push_back(
          makeSequentialBlock(seq_idx, schema.user, schema.sequential[seq_idx],
                              schema.timestamp_col_name, state, col_nums,
                              for_training, schema.multi_class_delim));
    }
    
    for (uint32_t dense_seq_idx = 0; dense_seq_idx < schema.dense_sequential.size(); dense_seq_idx++) {
      input_blocks.push_back(
          makeDenseSequentialBlock(dense_seq_idx, schema.user, schema.dense_sequential[dense_seq_idx], schema.timestamp_col_name, state, col_nums, for_training));
    }

    return input_blocks;
  }

  static dataset::BlockPtr makeCategoricalBlock(
      const CategoricalPair& categorical, DataState& state,
      const ColumnNumberMap& col_nums, std::optional<char> delimiter) {
    const auto& [cat_col_name, n_classes] = categorical;
    auto& string_vocab = state.vocabs_by_column[cat_col_name];
    if (!string_vocab) {
      string_vocab = dataset::ThreadSafeVocabulary::make(n_classes);
    }
    return dataset::StringLookupCategoricalBlock::make(
        col_nums.at(cat_col_name), string_vocab, delimiter);
  }

  static dataset::TrendBlockPtr makeDenseSequentialBlock(
    uint32_t dense_sequential_block_id, const CategoricalPair& user,
    const DenseSequentialQuadruplet& dense_sequential,
    const std::string& timestamp_col_name, DataState& state,
    const ColumnNumberMap& col_nums, bool for_training
  ) {
    const auto& [user_col_name, _] = user;
    const auto& [qty_col_name, lookahead, lookback, period] = dense_sequential;
    
    auto& user_qty_history =
        state.count_histories_by_id[dense_sequential_block_id];
    // Reset history if for training to prevent test data from leaking in.
    if (!user_qty_history || for_training) {
      user_qty_history =
          dataset::CountHistoryIndex::makeDefault();
    }

    return dataset::TrendBlock::make(
        /* has_count_col= */ true, /* id_col= */ col_nums.at(user_col_name), 
        /* timestamp_col= */ col_nums.at(timestamp_col_name), 
        /* count_col= */ col_nums.at(qty_col_name),
        lookahead, lookback, period, user_qty_history);
  }

  // We pass in an ID because sequential blocks can corrupt each other's states.
  static dataset::BlockPtr makeSequentialBlock(
      uint32_t sequential_block_id, const CategoricalPair& user,
      const SequentialTriplet& sequential,
      const std::string& timestamp_col_name, DataState& state,
      const ColumnNumberMap& col_nums, bool for_training,
      std::optional<char> delimiter) {
    const auto& [user_col_name, n_unique_users] = user;
    auto& user_vocab = state.vocabs_by_column[user_col_name];
    if (!user_vocab) {
      user_vocab = dataset::ThreadSafeVocabulary::make(n_unique_users);
    }

    const auto& [item_col_name, n_unique_items, track_last_n] = sequential;
    auto& item_vocab = state.vocabs_by_column[item_col_name];
    if (!item_vocab) {
      item_vocab = dataset::ThreadSafeVocabulary::make(n_unique_items);
    }

    auto& user_item_history =
        state.history_collections_by_id[sequential_block_id];
    // Reset history if for training to prevent test data from leaking in.
    if (!user_item_history || for_training) {
      user_item_history =
          dataset::ItemHistoryCollection::make(n_unique_users, track_last_n);
    }

    return dataset::UserItemHistoryBlock::make(
        col_nums.at(user_col_name), col_nums.at(item_col_name),
        col_nums.at(timestamp_col_name), user_vocab, item_vocab,
        user_item_history, delimiter);
  }
};

}  // namespace thirdai::bolt::sequential_classifier