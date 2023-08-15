#include "State.h"
#include <data/src/transformations/MachLabel.h>
#include <dataset/src/mach/MachIndex.h>
#include <proto/state.pb.h>

namespace thirdai::data {

using dataset::ThreadSafeVocabulary;
using dataset::mach::MachIndex;

proto::data::ItemHistoryTracker trackerToProto(
    const ItemHistoryTracker& tracker) {
  proto::data::ItemHistoryTracker tracker_proto;

  for (const auto& [key, history] : tracker.trackers) {
    auto* history_proto = tracker_proto.add_trackers();
    history_proto->set_key(key);

    for (const auto& record : history) {
      auto* record_proto = history_proto->add_history();
      record_proto->set_item(record.item);
      record_proto->set_timestamp(record.timestamp);
    }
  }

  tracker_proto.set_last_timestamp(tracker.last_timestamp);

  return tracker_proto;
}

ItemHistoryTracker trackerFromProto(
    const proto::data::ItemHistoryTracker& tracker_proto) {
  ItemHistoryTracker tracker;

  for (const auto& history : tracker_proto.trackers()) {
    for (const auto& record : history.history()) {
      tracker.trackers[history.key()].emplace_back(record.item(),
                                                   record.timestamp());
    }
  }

  tracker.last_timestamp = tracker_proto.last_timestamp();

  return tracker;
}

State::State(const proto::data::State& state_proto) {
  if (state_proto.has_mach_index()) {
    _mach_index = MachIndex::fromProto(state_proto.mach_index());
  }

  for (const auto& [vocab_key, vocab] : state_proto.vocabs()) {
    _vocabs[vocab_key] = ThreadSafeVocabulary::fromProto(vocab);
  }

  for (const auto& [tracker_key, tracker] :
       state_proto.item_history_trackers()) {
    _item_history_trackers[tracker_key] = trackerFromProto(tracker);
  }
}

proto::data::State* State::toProto() const {
  proto::data::State* state = new proto::data::State();

  if (_mach_index) {
    state->set_allocated_mach_index(_mach_index->toProto());
  }

  for (const auto& [vocab_key, vocab] : _vocabs) {
    state->mutable_vocabs()->emplace(vocab_key, vocab->toProto());
  }

  for (const auto& [tracker_key, tracker] : _item_history_trackers) {
    state->mutable_item_history_trackers()->emplace(tracker_key,
                                                    trackerToProto(tracker));
  }

  return state;
}

}  // namespace thirdai::data