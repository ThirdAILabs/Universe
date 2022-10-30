#pragma once

#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <deque>
#include <stdexcept>
#include <string>

namespace thirdai::dataset {

struct ItemRecord {
 public:
  ItemRecord(uint32_t item_id, int64_t timestamp)
      : _item_id(item_id), _timestamp(timestamp) {}

  uint32_t itemId() const { return _item_id; }

  int64_t timestamp() const { return _timestamp; }

 private:
  uint32_t _item_id;
  int64_t _timestamp;
};

class InteractionHistory {
 public:
  void addItems(int64_t timestamp,
                const ArrayColumn<uint32_t>::RowReference& items) {
    for (uint32_t item : items) {
      _history.emplace_back(item, timestamp);
    }
  }

  std::vector<uint32_t> getLastN(int64_t timestamp, uint32_t n,
                                 bool remove_stale_entries) {
    std::vector<uint32_t> last_n;

    uint32_t non_stale_items = 0;
    for (auto record = _history.rbegin(); record != _history.rend(); ++record) {
      if (record->timestamp() <= timestamp && last_n.size() < n) {
        last_n.push_back(record->itemId());
        if (last_n.size() == n) {
          break;
        }
      }
      // Any item that is within the last k of the given timestamp is not stale.
      non_stale_items++;
    }

    if (remove_stale_entries) {
      _history.erase(_history.begin(), _history.end() - non_stale_items);
    }

    return last_n;
  }

 private:
  std::deque<ItemRecord> _history;
};

class InteractionHistoryCollection {
 public:
  explicit InteractionHistoryCollection(uint32_t n_histories)
      : _histories(n_histories) {}

  InteractionHistory& at(uint32_t history_idx) {
    checkHistoryIndexValid(history_idx);
    return _histories[history_idx];
  }

  const InteractionHistory& at(uint32_t history_idx) const {
    checkHistoryIndexValid(history_idx);
    return _histories[history_idx];
  }

 private:
  void checkHistoryIndexValid(uint32_t history_idx) const {
    if (history_idx >= _histories.size()) {
      throw std::invalid_argument("No InteractionHistory for id: " +
                                  std::to_string(history_idx));
    }
  }

  std::vector<InteractionHistory> _histories;
};

using InteractionHistoryCollectionPtr =
    std::shared_ptr<InteractionHistoryCollection>;

class CategoricalTemporalTransformation final : public Transformation {
 public:
  CategoricalTemporalTransformation(std::string user_id_column_name,
                                    std::string item_id_column_name,
                                    std::string timestamp_column_name,
                                    std::string output_column_name,
                                    InteractionHistoryCollectionPtr history,
                                    uint32_t track_last_n, bool update_history,
                                    bool include_current_row, int64_t time_lag)
      : _user_id_column_name(std::move(user_id_column_name)),
        _item_column_name(std::move(item_id_column_name)),
        _timestamp_column_name(std::move(timestamp_column_name)),
        _output_column_name(std::move(output_column_name)),
        _history(std::move(history)),
        _track_last_n(track_last_n),
        _update_history(update_history),
        _include_current_row(include_current_row),
        _time_lag(time_lag) {}

  void apply(ColumnMap& columns) final;

 private:
  std::string _user_id_column_name;
  std::string _item_column_name;
  std::string _timestamp_column_name;
  std::string _output_column_name;

  InteractionHistoryCollectionPtr _history;

  uint32_t _track_last_n;
  bool _update_history;
  bool _include_current_row;
  int64_t _time_lag;
};

}  // namespace thirdai::dataset
