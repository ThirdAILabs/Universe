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
  ItemHistoryCollection(uint32_t n_histories, uint32_t max_items_per_history)
      : _max_items_per_history(max_items_per_history),
        _histories(n_histories) {}

  void add(uint32_t history_id, uint32_t item_id, uint32_t timestamp) {
    _histories.at(history_id).push_back({item_id, timestamp});
    while (_histories.at(history_id).size() > _max_items_per_history) {
      _histories.at(history_id).pop_front();
    }
  }

  uint32_t numHistories() const { return _histories.size(); }

  uint32_t maxItemsPerHistory() const { return _max_items_per_history; }

  const auto& at(uint32_t history_id) { return _histories.at(history_id); }

  static std::shared_ptr<ItemHistoryCollection> make(
      uint32_t n_histories, uint32_t max_items_per_history) {
    return std::make_shared<ItemHistoryCollection>(n_histories,
                                                   max_items_per_history);
  }

 private:
  uint32_t _max_items_per_history;
  std::vector<std::deque<ItemRecord>> _histories;

  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  ItemHistoryCollection() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_max_items_per_history, _histories);
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
                       ItemHistoryCollectionPtr item_history_collection)
      : _user_col(user_col),
        _item_col(item_col),
        _timestamp_col(timestamp_col),
        _track_last_n(item_history_collection->maxItemsPerHistory()),
        _user_id_lookup(std::move(user_id_map)),
        _item_id_lookup(std::move(item_id_map)),
        _records(std::move(item_history_collection)) {
    if (_user_id_lookup->vocabSize() > _records->numHistories()) {
      std::stringstream error_ss;
      error_ss << "[UserItemHistoryBlock] Invoked with incompatible "
                  "user_id_map and item_history_collection. There are "
               << _user_id_lookup->vocabSize()
               << " users in user_id_map "
                  "but item_history_collection only has enough space for "
               << _records->numHistories() << " users.";
      throw std::invalid_argument(error_ss.str());
    }
  }

  UserItemHistoryBlock(uint32_t user_col, uint32_t item_col,
                       uint32_t timestamp_col, uint32_t track_last_n,
                       uint32_t n_unique_users, uint32_t n_unique_items)
      : _user_col(user_col),
        _item_col(item_col),
        _timestamp_col(timestamp_col),
        _track_last_n(track_last_n),
        _user_id_lookup(ThreadSafeVocabulary::make(n_unique_users)),
        _item_id_lookup(ThreadSafeVocabulary::make(n_unique_items)),
        _records(ItemHistoryCollection::make(n_unique_users, track_last_n)) {}

  uint32_t featureDim() const final { return _item_id_lookup->vocabSize(); }

  bool isDense() const final { return false; }

  uint32_t expectedNumColumns() const final {
    uint32_t max_col_idx = std::max(_user_col, _item_col);
    max_col_idx = std::max(max_col_idx, _timestamp_col);
    return max_col_idx + 1;
  }

  static UserItemHistoryBlockPtr make(uint32_t user_col, uint32_t item_col,
                                      uint32_t timestamp_col,
                                      ThreadSafeVocabularyPtr user_id_map,
                                      ThreadSafeVocabularyPtr item_id_map,
                                      ItemHistoryCollectionPtr records) {
    return std::make_shared<UserItemHistoryBlock>(
        user_col, item_col, timestamp_col, std::move(user_id_map),
        std::move(item_id_map), std::move(records));
  }

  static UserItemHistoryBlockPtr make(uint32_t user_col, uint32_t item_col,
                                      uint32_t timestamp_col,
                                      uint32_t track_last_n,
                                      uint32_t n_unique_users,
                                      uint32_t n_unique_items) {
    return std::make_shared<UserItemHistoryBlock>(
        user_col, item_col, timestamp_col, track_last_n, n_unique_users,
        n_unique_items);
  }

  // have to see which one to send in case of multiple columns.
  uint32_t getColumnNum() const final { return _item_col; }

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    try {
      auto user_str = std::string(input_row.at(_user_col));
      auto item_str = std::string(input_row.at(_item_col));
      auto timestamp_str = std::string(input_row.at(_timestamp_col));

      uint32_t user_id = _user_id_lookup->getUid(user_str);
      int64_t epoch_timestamp = TimeObject(timestamp_str).secondsSinceEpoch();

      auto item_id = _item_id_lookup->getUid(item_str);

#pragma omp critical(user_item_history_block)
      {
        encodeTrackedItems(user_id, epoch_timestamp, vec);
        _records->add(user_id, item_id, epoch_timestamp);
      }

    } catch (...) {
      return std::current_exception();
    }
    return nullptr;
  }

 private:
  void encodeTrackedItems(uint32_t user_id, int64_t epoch_timestamp,
                          SegmentedFeatureVector& vec) {
    uint32_t added = 0;

    for (const auto& item : _records->at(user_id)) {
      if (added >= _track_last_n) {
        break;
      }

      if (item.timestamp <= epoch_timestamp) {
        vec.addSparseFeatureToSegment(item.item, 1.0);
        added++;
      }
    }
  }

  uint32_t _user_col;
  uint32_t _item_col;
  uint32_t _timestamp_col;
  uint32_t _track_last_n;

  ThreadSafeVocabularyPtr _user_id_lookup;
  ThreadSafeVocabularyPtr _item_id_lookup;

  std::shared_ptr<ItemHistoryCollection> _records;
};

}  // namespace thirdai::dataset