#pragma once

#include <dataset/src/DataLoader.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/encodings/categorical/StringLookup.h>
#include <dataset/src/encodings/categorical/ThreadSafeVocabulary.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::bolt::classifiers::sequential {

using CategoricalPair = std::pair<std::string, uint32_t>;
using SequentialTriplet = std::tuple<std::string, uint32_t, uint32_t>;

struct CategoricalFeat {
  std::string col_name;
  uint32_t vocab_size;

  CategoricalFeat(std::string col_name, uint32_t vocab_size)
      : col_name(std::move(col_name)), vocab_size(vocab_size) {}

  static CategoricalFeat fromPair(const CategoricalPair& cat_pair) {
    const auto& [col_name, vocab_size] = cat_pair;
    return {col_name, vocab_size};
  }
};

struct SequentialFeat {
  CategoricalFeat user;
  CategoricalFeat item;
  std::string timestamp_col_name;
  uint32_t track_last_n;

  static SequentialFeat fromPrimitives(
      const CategoricalPair& user_cat_pair,
      const SequentialTriplet& item_seq_triplet,
      const std::string& timestamp_col_name) {
    auto user = CategoricalFeat::fromPair(user_cat_pair);
    const auto& [col_name, vocab_size, track_last_n] = item_seq_triplet;
    auto item = CategoricalFeat(col_name, vocab_size);
    return {std::move(user), std::move(item), timestamp_col_name, track_last_n};
  }
};

struct Schema {
  Schema(const CategoricalPair& user, const CategoricalPair& target,
         const std::string& timestamp,
         std::vector<std::string> static_text = {},
         const std::vector<CategoricalPair>& static_categorical = {},
         const std::vector<SequentialTriplet>& sequential = {})
      : user(CategoricalFeat::fromPair(user)),
        target(CategoricalFeat::fromPair(target)),
        timestamp_col_name(timestamp),
        static_text_attrs(std::move(static_text)) {
    for (const auto& cat : static_categorical) {
      static_categorical_attrs.push_back(CategoricalFeat::fromPair(cat));
    }
    for (const auto& seq : sequential) {
      sequential_attrs.push_back(
          SequentialFeat::fromPrimitives(user, seq, timestamp));
    }
  }

  CategoricalFeat user;
  CategoricalFeat target;
  std::string timestamp_col_name;
  std::vector<std::string> static_text_attrs;
  std::vector<CategoricalFeat> static_categorical_attrs;
  std::vector<SequentialFeat> sequential_attrs;
};

struct DataState {
  std::unordered_map<std::string, dataset::ThreadSafeVocabularyPtr> vocabs;
  std::unordered_map<std::string, dataset::ItemHistoryCollectionPtr>
      history_collections;
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

  uint32_t at(const std::string& col_name) const {
    if (_name_to_num.count(col_name) == 0) {
      std::stringstream error_ss;
      error_ss << "Expected a column named '" << col_name
               << "' in header but could not find it.";
      throw std::runtime_error(error_ss.str());
    }
    return _name_to_num.at(col_name);
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

    auto header = file_reader->getHeader();
    if (!header) {
      throw std::runtime_error("File header not found.");
    }

    ColumnNumberMap col_nums(*header, delimiter);

    auto input_blocks = buildInputBlocks(schema, state, col_nums);

    std::vector<dataset::BlockPtr> label_blocks;
    label_blocks.push_back(
        makeCategoricalBlock(schema.target, state, col_nums));

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

 private:
  static std::vector<dataset::BlockPtr> buildInputBlocks(
      const Schema& schema, DataState& state, const ColumnNumberMap& col_nums) {
    std::vector<dataset::BlockPtr> input_blocks;
    input_blocks.push_back(makeCategoricalBlock(schema.user, state, col_nums));

    input_blocks.push_back(std::make_shared<dataset::DateBlock>(
        col_nums.at(schema.timestamp_col_name)));

    for (const auto& text_col_name : schema.static_text_attrs) {
      input_blocks.push_back(std::make_shared<dataset::TextBlock>(
          col_nums.at(text_col_name), /* dim = */ 100000));
    }

    for (const auto& categorical : schema.static_categorical_attrs) {
      input_blocks.push_back(
          makeCategoricalBlock(categorical, state, col_nums));
    }

    for (const auto& sequential : schema.sequential_attrs) {
      input_blocks.push_back(makeSequentialBlock(sequential, state, col_nums));
    }

    return input_blocks;
  }

  static dataset::BlockPtr makeCategoricalBlock(
      const CategoricalFeat& categorical, DataState& state,
      const ColumnNumberMap& col_nums) {
    auto& string_lookup = state.vocabs[categorical.col_name];
    if (!string_lookup) {
      string_lookup =
          dataset::ThreadSafeVocabulary::make(categorical.vocab_size);
    }
    return dataset::CategoricalBlock::make(
        col_nums.at(categorical.col_name),
        dataset::StringLookup::make(string_lookup));
  }

  static dataset::BlockPtr makeSequentialBlock(
      const SequentialFeat& sequential, DataState& state,
      const ColumnNumberMap& col_nums) {
    auto& user_lookup = state.vocabs[sequential.user.col_name];
    if (!user_lookup) {
      user_lookup =
          dataset::ThreadSafeVocabulary::make(sequential.user.vocab_size);
    }

    auto& item_lookup = state.vocabs[sequential.item.col_name];
    if (!item_lookup) {
      item_lookup =
          dataset::ThreadSafeVocabulary::make(sequential.item.vocab_size);
    }

    auto collection_name =
        sequential.item.col_name + std::to_string(sequential.track_last_n);
    auto& user_item_history = state.history_collections[collection_name];
    if (!user_item_history) {
      user_item_history = dataset::ItemHistoryCollection::make(
          sequential.user.vocab_size, sequential.track_last_n);
    }

    return dataset::UserItemHistoryBlock::make(
        col_nums.at(sequential.user.col_name),
        col_nums.at(sequential.item.col_name),
        col_nums.at(sequential.timestamp_col_name), user_lookup, item_lookup,
        user_item_history);
  }
};

}  // namespace thirdai::bolt::classifiers::sequential