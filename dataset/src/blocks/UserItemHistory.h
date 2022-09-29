#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <dataset/src/utils/TimeUtils.h>
#include <algorithm>
#include <atomic>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::dataset {

struct ItemRecord {
  uint32_t item;
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
  explicit ItemHistoryCollection(uint32_t n_histories)
      : _histories(n_histories) {}

  uint32_t numHistories() const { return _histories.size(); }

  auto& at(uint32_t history_id) { return _histories.at(history_id); }

  static std::shared_ptr<ItemHistoryCollection> make(uint32_t n_histories) {
    return std::make_shared<ItemHistoryCollection>(n_histories);
  }

 private:
  std::vector<std::deque<ItemRecord>> _histories;

  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  ItemHistoryCollection() {}

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
 public:
  UserItemHistoryBlock(uint32_t user_col, uint32_t item_col,
                       uint32_t timestamp_col,
                       ThreadSafeVocabularyPtr user_id_map,
                       ThreadSafeVocabularyPtr item_id_map,
                       ItemHistoryCollectionPtr item_history_collection,
                       uint32_t track_last_n, bool should_update_history = true,
                       bool include_current_row = false,
                       std::optional<char> item_col_delimiter = std::nullopt,
                       int64_t time_lag = 0)
      : _user_col(user_col),
        _item_col(item_col),
        _timestamp_col(timestamp_col),
        _track_last_n(track_last_n),
        _user_id_lookup(std::move(user_id_map)),
        _item_id_lookup(std::move(item_id_map)),
        _per_user_history(std::move(item_history_collection)),
        _should_update_history(should_update_history),
        _include_current_row(include_current_row),
        _item_col_delimiter(item_col_delimiter),
        _time_lag(time_lag) {
    if (_user_id_lookup->vocabSize() > _per_user_history->numHistories()) {
      std::stringstream error_ss;
      error_ss << "[UserItemHistoryBlock] Invoked with incompatible "
                  "user_id_map and item_history_collection. There are "
               << _user_id_lookup->vocabSize()
               << " users in user_id_map "
                  "but item_history_collection only has enough space for "
               << _per_user_history->numHistories() << " users.";
      throw std::invalid_argument(error_ss.str());
    }
  }

  UserItemHistoryBlock(uint32_t user_col, uint32_t item_col,
                       uint32_t timestamp_col, uint32_t track_last_n,
                       uint32_t n_unique_users, uint32_t n_unique_items,
                       bool should_update_history = true,
                       bool include_current_row = false,
                       std::optional<char> item_col_delimiter = std::nullopt,
                       int64_t time_lag = 0)
      : _user_col(user_col),
        _item_col(item_col),
        _timestamp_col(timestamp_col),
        _track_last_n(track_last_n),
        _user_id_lookup(ThreadSafeVocabulary::make(n_unique_users)),
        _item_id_lookup(ThreadSafeVocabulary::make(n_unique_items)),
        _per_user_history(ItemHistoryCollection::make(n_unique_users)),
        _should_update_history(should_update_history),
        _include_current_row(include_current_row),
        _item_col_delimiter(item_col_delimiter),
        _time_lag(time_lag) {}

  uint32_t featureDim() const final { return _item_id_lookup->vocabSize(); }

  bool isDense() const final { return false; }

  uint32_t expectedNumColumns() const final {
    uint32_t max_col_idx = std::max(_user_col, _item_col);
    max_col_idx = std::max(max_col_idx, _timestamp_col);
    return max_col_idx + 1;
  }

  static UserItemHistoryBlockPtr make(
      uint32_t user_col, uint32_t item_col, uint32_t timestamp_col,
      ThreadSafeVocabularyPtr user_id_map, ThreadSafeVocabularyPtr item_id_map,
      ItemHistoryCollectionPtr records, uint32_t track_last_n,
      bool should_update_history = true, bool include_current_row = false,
      std::optional<char> item_col_delimiter = std::nullopt,
      int64_t time_lag = 0) {
    return std::make_shared<UserItemHistoryBlock>(
        user_col, item_col, timestamp_col, std::move(user_id_map),
        std::move(item_id_map), std::move(records), track_last_n,
        should_update_history, include_current_row, item_col_delimiter,
        time_lag);
  }

  static UserItemHistoryBlockPtr make(
      uint32_t user_col, uint32_t item_col, uint32_t timestamp_col,
      uint32_t track_last_n, uint32_t n_unique_users, uint32_t n_unique_items,
      bool should_update_history = true, bool include_current_row = false,
      std::optional<char> item_col_delimiter = std::nullopt,
      int64_t time_lag = 0) {
    return std::make_shared<UserItemHistoryBlock>(
        user_col, item_col, timestamp_col, track_last_n, n_unique_users,
        n_unique_items, should_update_history, include_current_row,
        item_col_delimiter, time_lag);
  }

  // TODO(YASH): See whether length of history makes sense in explanations.
  Explanation explainIndex(
      uint32_t index_within_block,
      const std::vector<std::string_view>& input_row) final {
    (void)input_row;
    return {_item_col, "Previously seen '" +
                           _item_id_lookup->getString(index_within_block) +
                           "'"};
  }

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    try {
      auto user_str = std::string(input_row.at(_user_col));
      auto item_str = std::string(input_row.at(_item_col));
      auto timestamp_str = std::string(input_row.at(_timestamp_col));

      uint32_t user_id = _user_id_lookup->getUid(user_str);
      int64_t timestamp_seconds = TimeObject(timestamp_str).secondsSinceEpoch();

      std::vector<uint32_t> item_ids;
      if (!item_str.empty()) {
        item_ids = getItemIds(item_str);
      }

#pragma omp critical(user_item_history_block)
      {
        if (_include_current_row) {
          addNewItemsToUserHistory(user_id, timestamp_seconds, item_ids);
        }

        extendVectorWithUserHistory(
            user_id, timestamp_seconds - _time_lag, vec,
            /* remove_outdated_elements= */ _should_update_history);

        if (_include_current_row && !_should_update_history) {
          removeNewItemsFromUserHistory(user_id, item_ids);
        }

        if (!_include_current_row && _should_update_history) {
          addNewItemsToUserHistory(user_id, timestamp_seconds, item_ids);
        }
      }
    } catch (...) {
      return std::current_exception();
    }
    return nullptr;
  }

 private:
  std::vector<uint32_t> getItemIds(const std::string& item_str) {
    if (!_item_col_delimiter) {
      return {_item_id_lookup->getUid(item_str)};
    }
    auto item_id_views =
        ProcessorUtils::parseCsvRow(item_str, _item_col_delimiter.value());
    std::vector<uint32_t> item_id_strs;
    item_id_strs.reserve(item_id_views.size());
    for (auto item_id_view : item_id_views) {
      auto item_id_str = std::string(item_id_view);
      auto item_id = _item_id_lookup->getUid(item_id_str);
      item_id_strs.push_back(item_id);
    }
    return item_id_strs;
  }

  void extendVectorWithUserHistory(uint32_t user_id, int64_t timestamp_seconds,
                                   SegmentedFeatureVector& vec,
                                   bool remove_outdated_elements) {
    uint32_t seen = 0;
    uint32_t added = 0;

    auto& user_history = _per_user_history->at(user_id);

    for (auto record = user_history.rbegin(); record != user_history.rend();
         record++) {
      if (record->timestamp <= timestamp_seconds) {
        vec.addSparseFeatureToSegment(record->item, 1.0);
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

  // Returns elements removed from the item history in chronological order.
  void addNewItemsToUserHistory(uint32_t user_id, int64_t timestamp_seconds,
                                std::vector<uint32_t>& new_item_ids) {
    std::vector<ItemRecord> removed_elements;
    for (const auto& item_id : new_item_ids) {
      _per_user_history->at(user_id).push_back({item_id, timestamp_seconds});
    }
  }

  void removeNewItemsFromUserHistory(uint32_t user_id,
                                     std::vector<uint32_t>& new_item_ids) {
    auto& user_history = _per_user_history->at(user_id);
    user_history.erase(user_history.begin(),
                       user_history.begin() + new_item_ids.size());
  }

  uint32_t _user_col;
  uint32_t _item_col;
  uint32_t _timestamp_col;
  uint32_t _track_last_n;

  ThreadSafeVocabularyPtr _user_id_lookup;
  ThreadSafeVocabularyPtr _item_id_lookup;

  std::shared_ptr<ItemHistoryCollection> _per_user_history;

  bool _should_update_history;
  bool _include_current_row;
  std::optional<char> _item_col_delimiter;
  int64_t _time_lag;
};

}  // namespace thirdai::dataset