#pragma once

#include <dataset/src/DataLoader.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/encodings/categorical/StreamingStringCategoricalEncoding.h>
#include <dataset/src/encodings/categorical/StreamingStringLookup.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::bolt {

struct Sequential {
  struct Schema;
  struct State;
  class ColumnNumberMap;
  class Pipeline;
};

struct Sequential::Schema {
  struct Categorical {
    std::string col_name;
    uint32_t vocab_size;
  };

  struct Sequential {
    Categorical user;
    Categorical item;
    std::string timestamp_col_name;
    uint32_t track_last_n;
  };

  Categorical user;
  Categorical target;
  std::vector<std::string> static_text_attrs;
  std::vector<Categorical> static_categorical_attrs;
  std::vector<Sequential> sequential_attrs;
};

struct Sequential::State {
  using StringLookups =
      std::unordered_map<std::string, dataset::StreamingStringLookupPtr>;

  using UserItemHistories =
      std::unordered_map<std::string, dataset::UserItemHistoryRecordsPtr>;

  StringLookups lookups;
  UserItemHistories histories;
};

class Sequential::ColumnNumberMap {
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

class Sequential::Pipeline {
 public:
  static constexpr uint32_t BATCH_SIZE = 2048;

  static dataset::StreamingGenericDatasetLoader buildForFile(
      const Schema& schema, State& state, const std::string& filename,
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
    if (for_training) {
      label_blocks.push_back(
          makeCategoricalBlock(schema.target, state.lookups, col_nums));
    }

    return {file_reader,
            input_blocks,
            label_blocks,
            /* shuffle = */ for_training,
            /* config = */ {},
            /* has_header = */ false,  // since we already took the header.
            delimiter};
  }

 private:
  static std::vector<dataset::BlockPtr> buildInputBlocks(
      const Schema& schema, State& state, const ColumnNumberMap& col_nums) {
    std::vector<dataset::BlockPtr> input_blocks;

    input_blocks.push_back(
        makeCategoricalBlock(schema.user, state.lookups, col_nums));

    for (const auto& text_col_name : schema.static_text_attrs) {
      input_blocks.push_back(std::make_shared<dataset::TextBlock>(
          col_nums.at(text_col_name), /* dim = */ 100000));
    }

    for (const auto& categorical : schema.static_categorical_attrs) {
      input_blocks.push_back(
          makeCategoricalBlock(categorical, state.lookups, col_nums));
    }

    for (const auto& sequential : schema.sequential_attrs) {
      input_blocks.push_back(makeSequentialBlock(sequential, state.lookups,
                                                 state.histories, col_nums));
    }

    return input_blocks;
  }

  static dataset::BlockPtr makeCategoricalBlock(
      const Schema::Categorical& categorical, State::StringLookups& lookups,
      const ColumnNumberMap& col_nums) {
    auto& string_lookup = lookups.at(categorical.col_name);
    if (!string_lookup) {
      string_lookup = std::make_shared<dataset::StreamingStringLookup>(
          categorical.vocab_size);
    }
    return std::make_shared<dataset::CategoricalBlock>(
        col_nums.at(categorical.col_name),
        std::make_shared<dataset::StreamingStringCategoricalEncoding>(
            string_lookup));
  }

  static dataset::BlockPtr makeSequentialBlock(
      const Schema::Sequential& sequential, State::StringLookups& lookups,
      State::UserItemHistories& histories, const ColumnNumberMap& col_nums) {
    auto& user_lookup = lookups.at(sequential.user.col_name);
    if (!user_lookup) {
      user_lookup = std::make_shared<dataset::StreamingStringLookup>(
          sequential.user.vocab_size);
    }

    auto& item_lookup = lookups.at(sequential.item.col_name);
    if (!item_lookup) {
      item_lookup = std::make_shared<dataset::StreamingStringLookup>(
          sequential.item.vocab_size);
    }

    auto& user_item_history = histories.at(sequential.item.col_name);
    if (!user_item_history) {
      user_item_history = dataset::UserItemHistoryBlock::makeEmptyRecord(
          sequential.user.vocab_size, sequential.track_last_n);
    }

    return std::make_shared<dataset::UserItemHistoryBlock>(
        col_nums.at(sequential.user.col_name),
        col_nums.at(sequential.item.col_name),
        col_nums.at(sequential.timestamp_col_name), sequential.track_last_n,
        user_lookup, item_lookup, user_item_history);
  }
};

}  // namespace thirdai::bolt