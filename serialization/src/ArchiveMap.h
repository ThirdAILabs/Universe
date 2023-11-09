#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include <hashing/src/MurmurHash.h>
#include <serialization/src/Archive.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::ar {

class ArchiveMap final : public Archive {
 public:
  static std::shared_ptr<ArchiveMap> make() {
    return std::make_shared<ArchiveMap>();
  }

  bool contains(const std::string& key) const final {
    return _map.count(hash(key));
  }

  const ConstArchivePtr& get(const std::string& key) const final {
    if (contains(key)) {
      return _map.at(hash(key));
    }
    throw std::out_of_range("Map contains no value for key '" + key + "'.");
  }

  ConstArchivePtr& at(const std::string& key) { return _map[hash(key)]; }

  size_t size() const { return _map.size(); }

  auto begin() const { return _map.begin(); }

  auto end() const { return _map.end(); }

  std::string type() const final { return "Map"; }

 private:
  static uint64_t hash(const std::string& key) {
    // Murmur hash only outputs 32 bits.
    uint64_t h1 = hashing::MurmurHash(key.data(), key.size(), 89295);
    uint64_t h2 = hashing::MurmurHash(key.data(), key.size(), 34072);
    return (h1 << 32) + h2;
  }

  std::unordered_map<uint64_t, ConstArchivePtr> _map;

  friend class cereal::access;

  template <class Ar>
  void save(Ar& archive) const {
    archive(cereal::base_class<Archive>(this), _map);
  }

  template <class Ar>
  void load(Ar& archive) {
    archive(cereal::base_class<Archive>(this), _map);
  }
};

}  // namespace thirdai::ar

CEREAL_REGISTER_TYPE(thirdai::ar::ArchiveMap)
