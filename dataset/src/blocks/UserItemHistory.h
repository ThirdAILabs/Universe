#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/PreprocessedVectors.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <dataset/src/utils/TimeUtils.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {

struct ItemRecord {
  std::string item;
  int64_t timestamp;

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(item, timestamp);
  }
};

class ItemHistoryCollection {
 public:
  ItemHistoryCollection() {}

  uint32_t numHistories() const { return _histories.size(); }

  auto& at(const std::string& history_id) {
#pragma omp critical(item_history_collection)
    {
      if (!_histories.count(history_id)) {
        _histories[history_id];
      }
    }
    return _histories.at(history_id);
  }

  /**
   * Clears all tracked categories.
   */
  void reset() {
    for (auto& [id, history] : _histories) {
      history.clear();
    }
  }

  static std::shared_ptr<ItemHistoryCollection> make() {
    return std::make_shared<ItemHistoryCollection>();
  }

 private:
  std::unordered_map<std::string, std::deque<ItemRecord>> _histories;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_histories);
  }
};

using ItemHistoryCollectionPtr = std::shared_ptr<ItemHistoryCollection>;

class UserItemHistoryBlock;

using UserItemHistoryBlockPtr = std::shared_ptr<UserItemHistoryBlock>;

/**
 * Tracks up to the last N items associated with each user.
 */
class UserItemHistoryBlock final : public Block {
  static constexpr uint32_t ITEM_HASH_SEED = 341;

 public:
  UserItemHistoryBlock(uint32_t user_col, uint32_t item_col,
                       uint32_t timestamp_col,
                       ItemHistoryCollectionPtr item_history_collection,
                       uint32_t track_last_n, uint32_t item_hash_range,
                       bool should_update_history = true,
                       bool include_current_row = false,
                       std::optional<char> item_col_delimiter = std::nullopt,
                       int64_t time_lag = 0,
                       PreprocessedVectorsPtr item_vectors = nullptr)
      : _user_col(user_col),
        _item_col(item_col),
        _timestamp_col(timestamp_col),
        _track_last_n(track_last_n),
        _item_hash_range(item_hash_range),
        _item_vectors(std::move(item_vectors)),
        _per_user_history(std::move(item_history_collection)),
        _should_update_history(should_update_history),
        _include_current_row(include_current_row),
        _item_col_delimiter(item_col_delimiter),
        _time_lag(time_lag) {}

  uint32_t featureDim() const final {
    return _item_vectors ? _item_vectors->dim : _item_hash_range;
  }

  bool isDense() const final { return false; }

  uint32_t expectedNumColumns() const final {
    uint32_t max_col_idx = std::max(_user_col, _item_col);
    max_col_idx = std::max(max_col_idx, _timestamp_col);
    return max_col_idx + 1;
  }

  static UserItemHistoryBlockPtr make(
      uint32_t user_col, uint32_t item_col, uint32_t timestamp_col,
      ItemHistoryCollectionPtr records, uint32_t track_last_n,
      uint32_t item_hash_range, bool should_update_history = true,
      bool include_current_row = false,
      std::optional<char> item_col_delimiter = std::nullopt,
      int64_t time_lag = 0, PreprocessedVectorsPtr item_vectors = nullptr) {
    return std::make_shared<UserItemHistoryBlock>(
        user_col, item_col, timestamp_col, std::move(records), track_last_n,
        item_hash_range, should_update_history, include_current_row,
        item_col_delimiter, time_lag, std::move(item_vectors));
  }

  // TODO(YASH): See whether length of history makes sense in explanations.
  Explanation explainIndex(
      uint32_t index_within_block,
      const std::vector<std::string_view>& input_row) final {
    if (_item_vectors) {
      // TODO(Geordie): Make more descriptive.
      return {_item_col, "Metadata of previously seen item."};
    }

    auto user = std::string(input_row.at(_user_col));
    auto& user_history = _per_user_history->at(user);

    for (auto record = user_history.rbegin(); record != user_history.rend();
         record++) {
      if (getItemId(record->item) == index_within_block) {
        return {_item_col, "'" + record->item + "' is one of last " +
                               std::to_string(_track_last_n) + " values"};
      }
    }

    return {_item_col,
            "One of last " + std::to_string(_track_last_n) + " values"};
  }

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    try {
      auto user_str = std::string(input_row.at(_user_col));
      auto item_str = std::string(input_row.at(_item_col));
      auto timestamp_str = std::string(input_row.at(_timestamp_col));

      int64_t timestamp_seconds = TimeObject(timestamp_str).secondsSinceEpoch();

      std::vector<std::string> items;
      if (!item_str.empty()) {
        items = getItems(item_str);
      }

#pragma omp critical(user_item_history_block)
      {
        if (_include_current_row) {
          addNewItemsToUserHistory(user_str, timestamp_seconds, items);
        }

        extendVectorWithUserHistory(
            user_str, timestamp_seconds - _time_lag, vec,
            /* remove_outdated_elements= */ _should_update_history);

        if (_include_current_row && !_should_update_history) {
          removeNewItemsFromUserHistory(user_str, items);
        }

        if (!_include_current_row && _should_update_history) {
          addNewItemsToUserHistory(user_str, timestamp_seconds, items);
        }
      }
    } catch (...) {
      return std::current_exception();
    }
    return nullptr;
  }

 private:
  std::vector<std::string> getItems(const std::string& item_str) {
    if (!_item_col_delimiter) {
      return {item_str};
    }
    auto item_id_views =
        ProcessorUtils::parseCsvRow(item_str, _item_col_delimiter.value());
    std::vector<std::string> items;
    items.reserve(item_id_views.size());
    for (auto item_id_view : item_id_views) {
      items.push_back(std::string(item_id_view));
    }
    return items;
  }

  void extendVectorWithUserHistory(const std::string& user,
                                   int64_t timestamp_seconds,
                                   SegmentedFeatureVector& vec,
                                   bool remove_outdated_elements) {
    uint32_t seen = 0;
    uint32_t added = 0;

    auto& user_history = _per_user_history->at(user);

    for (auto record = user_history.rbegin(); record != user_history.rend();
         record++) {
      if (record->timestamp <= timestamp_seconds) {
        if (_item_vectors) {
          _item_vectors->appendPreprocessedFeaturesToVector(record->item, vec);
        } else {
          vec.addSparseFeatureToSegment(getItemId(record->item), 1.0);
        }
        added++;
      }
      seen++;

      if (added >= _track_last_n) {
        break;
      }
    }

    if (remove_outdated_elements) {
      uint32_t n_outdated = user_history.size() - seen;
      user_history.erase(user_history.begin(),
                         user_history.begin() + n_outdated);
    }
  }

  uint32_t getItemId(const std::string& item) const {
    return hashing::MurmurHash(item.data(), item.size(), ITEM_HASH_SEED) %
           _item_hash_range;
  }

  // Returns elements removed from the item history in chronological order.
  void addNewItemsToUserHistory(const std::string& user,
                                int64_t timestamp_seconds,
                                std::vector<std::string>& new_items) {
    std::vector<ItemRecord> removed_elements;
    for (const auto& item : new_items) {
      _per_user_history->at(user).push_back({item, timestamp_seconds});
    }
  }

  void removeNewItemsFromUserHistory(const std::string& user,
                                     std::vector<std::string>& new_items) {
    auto& user_history = _per_user_history->at(user);
    user_history.erase(user_history.end() - new_items.size(),
                       user_history.end());
  }

  uint32_t _user_col;
  uint32_t _item_col;
  uint32_t _timestamp_col;
  uint32_t _track_last_n;
  uint32_t _item_hash_range;

  PreprocessedVectorsPtr _item_vectors;

  std::shared_ptr<ItemHistoryCollection> _per_user_history;

  bool _should_update_history;
  bool _include_current_row;
  std::optional<char> _item_col_delimiter;
  int64_t _time_lag;

  // Constructor for Cereal
  UserItemHistoryBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _user_col, _item_col,
            _timestamp_col, _track_last_n, _item_hash_range, _item_vectors,
            _per_user_history, _should_update_history, _include_current_row,
            _item_col_delimiter, _time_lag);
  }
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::UserItemHistoryBlock)