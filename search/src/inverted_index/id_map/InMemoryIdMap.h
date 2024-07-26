#pragma once

#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <search/src/inverted_index/id_map/IdMap.h>
#include <memory>
#include <unordered_map>

namespace thirdai::search {

class InMemoryIdMap final : public IdMap {
 public:
  explicit InMemoryIdMap(
      std::unordered_map<uint64_t, std::vector<uint64_t>> key_to_values = {});

  std::vector<uint64_t> get(uint64_t key) const final {
    return _key_to_values.at(key);
  }

  void put(uint64_t key, const std::vector<uint64_t>& values) final;

  std::vector<uint64_t> deleteValue(uint64_t value) final;

  uint64_t maxKey() const final;

  void save(const std::string& path) const final;

  static std::unique_ptr<InMemoryIdMap> load(const std::string& path);

  std::string type() const final { return typeName(); }

  static std::string typeName() { return "in-memory"; }

 private:
  std::unordered_map<uint64_t, std::vector<uint64_t>> _key_to_values;
  std::unordered_map<uint64_t, std::vector<uint64_t>> _value_to_keys;
};

}  // namespace thirdai::search