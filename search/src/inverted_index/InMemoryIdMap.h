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
      std::unordered_map<uint64_t, std::vector<uint64_t>> map = {})
      : _map(std::move(map)) {}

  std::vector<uint64_t> get(uint64_t key) const final { return _map.at(key); }

  bool contains(uint64_t key) const final { return _map.count(key); }

  void put(uint64_t key, std::vector<uint64_t> value) final {
    _map[key] = value;
  }

  void append(uint64_t key, uint64_t value) final {
    _map[key].push_back(value);
  }

  void del(uint64_t key) final { _map.erase(key); }

  void save(const std::string& path) const final {
    auto archive = ar::Map::make();
    archive->set("map", ar::mapU64VecU64(_map));

    auto ofile = dataset::SafeFileIO::ofstream(path);

    ar::serialize(archive, ofile);
  }

  static std::unique_ptr<InMemoryIdMap> load(const std::string& path) {
    auto ifile = dataset::SafeFileIO::ifstream(path);
    auto archive = ar::deserialize(ifile);

    auto map = std::make_unique<InMemoryIdMap>(
        archive->getAs<ar::MapU64VecU64>("map"));

    return map;
  }

  std::string type() const final { return typeName(); }

  static std::string typeName() { return "in-memory"; }

 private:
  std::unordered_map<uint64_t, std::vector<uint64_t>> _map;
};

}  // namespace thirdai::search