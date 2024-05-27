#include "State.h"
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <stdexcept>
#include <vector>

namespace thirdai::data {

using dataset::ThreadSafeVocabulary;
using dataset::mach::MachIndex;

ar::ConstArchivePtr itemHistoryTrackerToArchive(
    const ItemHistoryTracker& history) {
  auto items = ar::Map::make();
  auto timestamps = ar::Map::make();

  for (const auto& [id, item_timestamps] : history) {
    std::vector<uint32_t> id_items;
    std::vector<int64_t> id_timestamps;

    for (const auto& [item, timestamp] : item_timestamps) {
      id_items.push_back(item);
      id_timestamps.push_back(timestamp);
    }

    items->set(id, ar::vecU32(std::move(id_items)));
    timestamps->set(id, ar::vecI64(std::move(id_timestamps)));
  }

  auto map = ar::Map::make();
  map->set("items", items);
  map->set("timestamps", timestamps);

  return map;
}

ar::ConstArchivePtr countHistoryTrackerToArchive(
    const CountHistoryTracker& history) {
  auto values = ar::Map::make();
  auto timestamps = ar::Map::make();

  for (const auto& [id, item_timestamps] : history) {
    std::vector<float> id_values;
    std::vector<int64_t> id_timestamps;

    for (const auto& [value, timestamp] : item_timestamps) {
      id_values.push_back(value);
      id_timestamps.push_back(timestamp);
    }

    values->set(id, ar::vecF32(std::move(id_values)));
    timestamps->set(id, ar::vecI64(std::move(id_timestamps)));
  }

  auto map = ar::Map::make();
  map->set("values", values);
  map->set("timestamps", timestamps);

  return map;
}

ar::ConstArchivePtr State::toArchive() const {
  auto map = ar::Map::make();

  if (_mach_index) {
    map->set("mach_index", _mach_index->toArchive());
  }

  if (_mach_memory) {
    map->set("mach_memory", _mach_memory->toArchive());
  }

  auto vocabs = ar::Map::make();
  for (const auto& [k, v] : _vocabs) {
    vocabs->set(k, v->toArchive());
  }
  map->set("vocabs", vocabs);

  auto item_trackers = ar::Map::make();
  for (const auto& [k, v] : _item_history_trackers) {
    item_trackers->set(k, itemHistoryTrackerToArchive(v));
  }
  map->set("item_history_trackers", item_trackers);

  auto count_trackers = ar::Map::make();
  for (const auto& [k, v] : _count_history_trackers) {
    count_trackers->set(k, countHistoryTrackerToArchive(v));
  }
  map->set("count_history_trackers", count_trackers);

  if (_graph) {
    map->set("graph", _graph->toArchive());
  }

  return map;
}

ItemHistoryTracker itemHistoryTrackerFromArchive(const ar::Archive& archive) {
  ItemHistoryTracker history;

  const auto& items = archive.get("items")->map();
  const auto& timestamps = archive.get("timestamps")->map();

  for (const auto& [id, ar_id_items] : items) {
    const auto& id_items = ar_id_items->as<ar::VecU32>();
    const auto& id_timestamps = timestamps.getAs<ar::VecI64>(id);

    if (id_items.size() != id_timestamps.size()) {
      throw std::invalid_argument(
          "Mismatch between item size and timestamp size deserializing history "
          "tracker.");
    }

    std::deque<ItemRecord> item_records;
    for (size_t i = 0; i < id_items.size(); i++) {
      item_records.push_back({id_items[i], id_timestamps[i]});
    }
    history[id] = std::move(item_records);
  }

  return history;
}

CountHistoryTracker countHistoryTrackerFromArchive(const ar::Archive& archive) {
  CountHistoryTracker history;

  const auto& values = archive.get("values")->map();
  const auto& timestamps = archive.get("timestamps")->map();

  for (const auto& [id, ar_id_values] : values) {
    const auto& id_items = ar_id_values->as<ar::VecF32>();
    const auto& id_timestamps = timestamps.getAs<ar::VecI64>(id);

    if (id_items.size() != id_timestamps.size()) {
      throw std::invalid_argument(
          "Mismatch between item size and timestamp size deserializing history "
          "tracker.");
    }

    std::deque<CountRecord> item_records;
    for (size_t i = 0; i < id_items.size(); i++) {
      item_records.push_back({id_items[i], id_timestamps[i]});
    }
    history[id] = std::move(item_records);
  }

  return history;
}

std::shared_ptr<State> State::fromArchive(const ar::Archive& archive) {
  return std::make_shared<State>(archive);
}

State::State(const ar::Archive& archive) {
  if (archive.contains("mach_index")) {
    _mach_index = MachIndex::fromArchive(*archive.get("mach_index"));
  }

  if (archive.contains("mach_memory")) {
    _mach_memory = MachMemory::fromArchive(*archive.get("mach_memory"));
  }

  for (const auto& [k, v] : archive.get("vocabs")->map()) {
    _vocabs[k] = ThreadSafeVocabulary::fromArchive(*v);
  }

  for (const auto& [k, v] : archive.get("item_history_trackers")->map()) {
    _item_history_trackers[k] = itemHistoryTrackerFromArchive(*v);
  }

  if (archive.contains("count_history_trackers")) {
    for (const auto& [k, v] : archive.get("count_history_trackers")->map()) {
      _count_history_trackers[k] = countHistoryTrackerFromArchive(*v);
    }
  }

  if (archive.contains("graph")) {
    _graph = automl::GraphInfo::fromArchive(*archive.get("graph"));
  }
}

}  // namespace thirdai::data