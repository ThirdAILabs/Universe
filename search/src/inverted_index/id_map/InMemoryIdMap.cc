#include "InMemoryIdMap.h"

namespace thirdai::search {

InMemoryIdMap::InMemoryIdMap(
    std::unordered_map<uint64_t, std::vector<uint64_t>> key_to_values)
    : _key_to_values(std::move(key_to_values)) {
  for (const auto& [key, values] : _key_to_values) {
    for (auto value : values) {
      _value_to_keys[value].push_back(key);
    }
  }
}

void InMemoryIdMap::put(uint64_t key, const std::vector<uint64_t>& values) {
  _key_to_values[key] = values;
  for (uint64_t value : values) {
    _value_to_keys[value].push_back(key);
  }
}

std::vector<uint64_t> InMemoryIdMap::deleteValue(uint64_t value) {
  if (!_value_to_keys.count(value)) {
    return {};
  }

  std::vector<uint64_t> empty_keys;
  for (const auto& key : _value_to_keys.at(value)) {
    auto& values = _key_to_values.at(key);

    auto loc = std::find(values.begin(), values.end(), value);
    if (loc != values.end()) {
      values.erase(loc);
    }
    if (values.empty()) {
      _key_to_values.erase(key);
      empty_keys.push_back(key);
    }
  }

  _value_to_keys.erase(value);

  return empty_keys;
}

uint64_t InMemoryIdMap::maxKey() const {
  uint64_t max_key = 0;
  for (const auto& [key, _] : _key_to_values) {
    if (key > max_key) {
      max_key = key;
    }
  }
  return max_key;
}

void InMemoryIdMap::save(const std::string& path) const {
  auto archive = ar::Map::make();
  archive->set("key_to_values", ar::mapU64VecU64(_key_to_values));

  auto ofile = dataset::SafeFileIO::ofstream(path);

  ar::serialize(archive, ofile);
}

std::unique_ptr<InMemoryIdMap> InMemoryIdMap::load(const std::string& path) {
  auto ifile = dataset::SafeFileIO::ifstream(path);
  auto archive = ar::deserialize(ifile);

  auto map = std::make_unique<InMemoryIdMap>(
      archive->getAs<ar::MapU64VecU64>("key_to_values"));

  return map;
}

}  // namespace thirdai::search