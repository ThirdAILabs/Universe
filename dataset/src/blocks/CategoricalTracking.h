#include <dataset/src/blocks/BlockInterface.h>
#include <algorithm>
#include <atomic>
#include <exception>
#include <unordered_map>
namespace thirdai::dataset {

template<typename ELEMENT_T>
class UserItemBuffer {
  
 public:
  explicit UserItemBuffer(size_t size) : _buffer(size), _idx(0) {}

  void insert(ELEMENT_T element) {
    _buffer[getAndUpdateIdx()] = std::move(element);
  }

  const std::vector<std::atomic<ELEMENT_T>>& view() { return _buffer; }

 private:
  uint32_t getAndUpdateIdx() {
    uint32_t cur_idx = _idx.load();
    uint32_t new_idx = nextIdx(cur_idx);
    while(!_idx.compare_exchange_weak(cur_idx, new_idx)) {
      new_idx = nextIdx(cur_idx);
    }
    return cur_idx;
  }

  uint32_t nextIdx(uint32_t idx) {
    return (idx + 1) % _buffer.size();
  }

  std::vector<std::atomic<ELEMENT_T>> _buffer;
  std::atomic_uint32_t _idx;
};

struct ItemRecord {
  uint32_t item;
  uint64_t timestamp;
};

/**
 * Tracks up to the last N items associated with each user.
 */
class PerUserItemTracking final : public Block {

 public:
  PerUserItemTracking(
    uint32_t user_col, uint32_t item_col, uint32_t timestamp_col, 
    std::shared_ptr<std::unordered_map<std::string, uint32_t>> user_id_map, 
    std::shared_ptr<std::unordered_map<std::string, uint32_t>> item_id_map, 
    std::shared_ptr<std::vector<UserItemBuffer<ItemRecord>>> records)
    : _user_col(user_col), _item_col(item_col), _timestamp_col(timestamp_col), _user_id_map(std::move(user_id_map)), _item_id_map(std::move(item_id_map)), _records(std::move(records)) {
  }

  uint32_t featureDim() const final {
    return _item_id_map->size();
  }

  bool isDense() const final {
    return false;
  }

  uint32_t expectedNumColumns() const final {
    uint32_t max_col_idx = std::max(_user_col, _item_col);
    max_col_idx = std::max(max_col_idx, _timestamp_col);
    return max_col_idx + 1;
  }

 protected:
  std::exception_ptr buildSegment(const std::vector<std::string_view> &input_row, SegmentedFeatureVector &vec) final {
    try {
      auto user = std::string(input_row.at(_user_col));
      uint32_t user_id = _user_id_map->at(user);
      
      auto item = std::string(input_row.at(_item_col));
      uint32_t item_id = _item_id_map->at(item);

      auto timestamp = std::string(input_row.at(_timestamp_col));
      uint32_t epoch_timestamp = Time

    } catch (std::exception_ptr& except) {
      return except;
    }
    return nullptr;
  }

 private:
  uint32_t _user_col;
  uint32_t _item_col;
  uint32_t _timestamp_col;

  std::shared_ptr<std::unordered_map<std::string, uint32_t>> _user_id_map;
  std::shared_ptr<std::unordered_map<std::string, uint32_t>> _item_id_map;

  std::shared_ptr<std::vector<UserItemBuffer<ItemRecord>>> _records;
};

} // namespace thirdai::dataset