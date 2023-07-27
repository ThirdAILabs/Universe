#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <dataset/src/mach/MachIndex.h>
#include <stdexcept>

namespace thirdai::data {

using dataset::mach::MachIndexPtr;

struct ItemRecord {
  uint32_t item;
  int64_t timestamp;
};

using ItemHistoryTracker =
    std::unordered_map<std::string, std::deque<ItemRecord>>;
using ItemHistoryTrackerPtr = std::shared_ptr<ItemHistoryTracker>;

class State {
 public:
  explicit State(MachIndexPtr mach_index)
      : _mach_index(std::move(mach_index)) {}

  State() {}

  const auto& machIndex() const {
    if (!_mach_index) {
      throw std::invalid_argument(
          "Transformation state does not contain MachIndex.");
    }
    return _mach_index;
  }

  void setMachIndex(MachIndexPtr new_index) {
    if (_mach_index->numBuckets() != new_index->numBuckets()) {
      throw std::invalid_argument(
          "Output range mismatch in new index. Index output range should be " +
          std::to_string(_mach_index->numBuckets()) +
          " but provided an index with range = " +
          std::to_string(new_index->numBuckets()) + ".");
    }

    _mach_index = std::move(new_index);
  }

  const ItemHistoryTrackerPtr& getItemHistoryTracker(
      const std::string& user_column, const std::string& item_column,
      const std::string& timestamp_column) {
    std::string name = user_column + "_" + item_column + "_" + timestamp_column;
    if (!_item_history_trackers.count(name)) {
      _item_history_trackers[name] = std::make_shared<ItemHistoryTracker>();
    }
    return _item_history_trackers.at(name);
  }

 private:
  MachIndexPtr _mach_index = nullptr;

  std::unordered_map<std::string, ItemHistoryTrackerPtr> _item_history_trackers;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_mach_index);
  }
};

}  // namespace thirdai::data