#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <dataset/src/mach/MachIndex.h>
#include <limits>
#include <stdexcept>

namespace thirdai::data {

using dataset::mach::MachIndexPtr;

struct ItemRecord {
  uint32_t item;
  int64_t timestamp;
};

struct ItemHistoryTracker {
  std::unordered_map<std::string, std::deque<ItemRecord>> trackers;
  int64_t last_timestamp = std::numeric_limits<int64_t>::min();
};

/**
 * The purpose of this state object is to have a central location where stateful
 * information is stored in the data pipeline. Having a unique owner for all the
 * stateful information simplifies the serialization because the information is
 * only serialized from, and deserialized to a single place. The state object is
 * passed to each transformation's apply method so that they can access any of
 * the information stored in the state.
 */
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

  ItemHistoryTracker& getItemHistoryTracker(const std::string& tracker_key) {
    return _item_history_trackers[tracker_key];
  }

 private:
  MachIndexPtr _mach_index = nullptr;

  std::unordered_map<std::string, ItemHistoryTracker> _item_history_trackers;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_mach_index);
  }
};

}  // namespace thirdai::data