#pragma once

#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/encodings/categorical/StreamingStringLookup.h>
#include <dataset/src/utils/TimeUtils.h>
#include <algorithm>
#include <atomic>
#include <exception>
#include <optional>
#include <unordered_map>
namespace thirdai::dataset {
struct ItemRecord {
  bool valid;
  uint32_t item;
  int64_t timestamp;
};

class UserItemBuffer {
 public:
  explicit UserItemBuffer(size_t size) : _buffer(size), _idx(0) {}

  UserItemBuffer(const UserItemBuffer& other)
      : _buffer(other._buffer), _idx(other._idx) {}

  void insert(ItemRecord element) { _buffer[getAndUpdateIdx()] = element; }

  const std::vector<ItemRecord>& view() { return _buffer; }

 private:
  uint32_t getAndUpdateIdx() {
    uint32_t cur_idx = _idx;
    _idx = nextIdx(cur_idx);
    return cur_idx;
  }

  uint32_t nextIdx(uint32_t idx) { return (idx + 1) % _buffer.size(); }

  std::vector<ItemRecord> _buffer;
  uint32_t _idx;
};

using UserItemHistoryRecords = std::vector<UserItemBuffer>;
using UserItemHistoryRecordsPtr = std::shared_ptr<UserItemHistoryRecords>;

/**
 * Tracks up to the last N items associated with each user.
 */
class UserItemHistoryBlock final : public Block {
 public:
  UserItemHistoryBlock(uint32_t user_col, uint32_t item_col,
                       uint32_t timestamp_col, uint32_t track_last_n,
                       std::shared_ptr<StreamingStringLookup> user_id_map,
                       std::shared_ptr<StreamingStringLookup> item_id_map,
                       std::shared_ptr<UserItemHistoryRecords> records)
      : _user_col(user_col),
        _item_col(item_col),
        _timestamp_col(timestamp_col),
        _track_last_n(track_last_n),
        _user_id_lookup(std::move(user_id_map)),
        _item_id_lookup(std::move(item_id_map)),
        _records(std::move(records)) {}

  static std::shared_ptr<UserItemHistoryRecords> makeEmptyRecord(
      uint32_t n_users, uint32_t track_last_n) {
    UserItemHistoryRecords records(n_users, UserItemBuffer(track_last_n));
    return std::make_shared<UserItemHistoryRecords>(std::move(records));
  }

  uint32_t featureDim() const final { return _item_id_lookup->vocabSize(); }

  bool isDense() const final { return false; }

  uint32_t expectedNumColumns() const final {
    uint32_t max_col_idx = std::max(_user_col, _item_col);
    max_col_idx = std::max(max_col_idx, _timestamp_col);
    return max_col_idx + 1;
  }

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    try {
      auto user_str = std::string(input_row.at(_user_col));
      auto item_str = std::string(input_row.at(_item_col));
      auto timestamp_str = std::string(input_row.at(_timestamp_col));

      uint32_t user_id = _user_id_lookup->lookup(user_str);
      uint32_t item_id = _item_id_lookup->lookup(item_str);
      int64_t epoch_timestamp = TimeObject(timestamp_str).secondsSinceEpoch();

#pragma omp critical(user_item_history_block)
      {
        encodeTrackedItems(user_id, epoch_timestamp, vec);

        // Insert new item after adding to the vector to not give away new
        _records->at(user_id).insert({/* valid = */ true, /* item = */ item_id,
                                      /* timestamp = */ epoch_timestamp});
      }

    } catch (std::exception& except) {
      return std::make_exception_ptr(except);
    }
    return nullptr;
  }

 private:
  void encodeTrackedItems(uint32_t user_id, int64_t epoch_timestamp,
                          SegmentedFeatureVector& vec) {
    uint32_t added = 0;

    for (const auto& item : _records->at(user_id).view()) {
      if (added >= _track_last_n) {
        break;
      }

      if (item.valid && item.timestamp <= epoch_timestamp) {
        vec.addSparseFeatureToSegment(item.item, 1.0);
        added++;
      }
    }
  }

  uint32_t _user_col;
  uint32_t _item_col;
  uint32_t _timestamp_col;
  uint32_t _track_last_n;

  std::shared_ptr<StreamingStringLookup> _user_id_lookup;
  std::shared_ptr<StreamingStringLookup> _item_id_lookup;

  std::shared_ptr<UserItemHistoryRecords> _records;
};

}  // namespace thirdai::dataset