#pragma once

#include <dataset/src/mach/MachIndex.h>
#include <dataset/src/utils/GraphInfo.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <proto/state.pb.h>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::data {

using dataset::ThreadSafeVocabularyPtr;
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
 *
 * Comment on design: We chose to store different types of state as explicit
 * fields instead of in a map with a state interface for the following reasons:
 *    1. No common behavior or properties makes strange to have a unifying
 *       interface for different types of state.
 *    2. This design simplifies using the state because it won't require a lot
 *       of dynamic casting to get the correct state types from the map.
 *    3. Similarly it reduces sources of error because the types are explicit
 *       and clear, there are fewer possible sources of error when fields of a
 *       particular type are accessed directly, rather than having to access,
 *       check types, and cast.
 *    4. It simplifies handling namespaces between different types of state,
 *       i.e. avoid key collisions. Especially for things like a Mach Index
 *       where we only need one for a given state object. With this design it
 *       can just be accessed directly instead of needing to be retrieved from a
 *       map.
 *    5. At least when this decision was made, there are only a few types of
 *       state we actually have, so having a field for each is not an issue.
 */
class State {
 public:
  explicit State(MachIndexPtr mach_index)
      : _mach_index(std::move(mach_index)) {}

  explicit State(automl::data::GraphInfoPtr graph) : _graph(std::move(graph)) {}

  State(MachIndexPtr mach_index, automl::data::GraphInfoPtr graph)
      : _mach_index(std::move(mach_index)), _graph(std::move(graph)) {}

  State() {}

  explicit State(const proto::data::State& state);

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

  bool containsVocab(const std::string& key) const {
    return _vocabs.count(key);
  }

  void addVocab(const std::string& key, ThreadSafeVocabularyPtr&& vocab) {
    _vocabs.emplace(key, std::move(vocab));
  }

  ThreadSafeVocabularyPtr& getVocab(const std::string& key) {
    if (!_vocabs.count(key)) {
      throw std::invalid_argument("Cannot find vocab for key '" + key + "'.");
    }
    return _vocabs.at(key);
  }

  ItemHistoryTracker& getItemHistoryTracker(const std::string& tracker_key) {
    return _item_history_trackers[tracker_key];
  }

  void clearHistoryTrackers() { _item_history_trackers.clear(); }

  const auto& graph() const {
    if (!_graph) {
      throw std::invalid_argument(
          "Transformation state does not contain Graph object.");
    }
    return _graph;
  }

  proto::data::State* toProto() const;

 private:
  MachIndexPtr _mach_index = nullptr;

  std::unordered_map<std::string, ThreadSafeVocabularyPtr> _vocabs;

  std::unordered_map<std::string, ItemHistoryTracker> _item_history_trackers;

  automl::data::GraphInfoPtr _graph;
};

using StatePtr = std::shared_ptr<State>;

}  // namespace thirdai::data