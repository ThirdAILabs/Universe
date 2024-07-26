#pragma once

#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <search/src/inverted_index/IdMap.h>
#include <memory>
#include <unordered_map>

namespace thirdai::search {

class InMemoryIdMap final : public IdMap {
 public:
  explicit InMemoryIdMap(
      std::unordered_map<uint64_t, std::vector<uint64_t>> key_to_values = {})
      : _key_to_values(std::move(key_to_values)) {
    for (const auto& [key, values] : _key_to_values) {
      for (auto value : values) {
        _value_to_keys[value].push_back(key);
      }
    }
  }

  std::vector<uint64_t> get(uint64_t key) const final {
    return _key_to_values.at(key);
  }

  void put(uint64_t key, std::vector<uint64_t> value) final {
    _key_to_values[key] = value;
  }

  std::vector<uint64_t> deleteValue(uint64_t value) final {
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

  void save(const std::string& path) const final {
    auto archive = ar::Map::make();
    archive->set("key_to_values", ar::mapU64VecU64(_key_to_values));

    auto ofile = dataset::SafeFileIO::ofstream(path);

    ar::serialize(archive, ofile);
  }

  static std::unique_ptr<InMemoryIdMap> load(const std::string& path) {
    auto ifile = dataset::SafeFileIO::ifstream(path);
    auto archive = ar::deserialize(ifile);

    auto map = std::make_unique<InMemoryIdMap>(
        archive->getAs<ar::MapU64VecU64>("key_to_values"));

    return map;
  }

  std::string type() const final { return typeName(); }

  static std::string typeName() { return "in-memory"; }

 private:
  std::unordered_map<uint64_t, std::vector<uint64_t>> _key_to_values;
  std::unordered_map<uint64_t, std::vector<uint64_t>> _value_to_keys;
};

}  // namespace thirdai::search