#pragma once

#include <cereal/types/polymorphic.hpp>
#include <hashing/src/MurmurHash.h>
#include <archive/src/Archive.h>
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

  bool contains(const std::string& key) const final { return _map.count(key); }

  const ConstArchivePtr& get(const std::string& key) const final {
    if (contains(key)) {
      return _map.at(key);
    }
    throw std::out_of_range("Map contains no value for key '" + key + "'.");
  }

  void set(const std::string& key, ConstArchivePtr archive) {
    if (contains(key)) {
      throw std::runtime_error("Found duplicate entry for key in Archive Map.");
    }
    _map[key] = std::move(archive);
  }

  size_t size() const { return _map.size(); }

  auto begin() const { return _map.begin(); }

  auto end() const { return _map.end(); }

  std::string type() const final { return "Map"; }

 private:
  std::unordered_map<std::string, ConstArchivePtr> _map;

  friend class cereal::access;

  template <class Ar>
  void save(Ar& archive) const;

  template <class Ar>
  void load(Ar& archive);
};

}  // namespace thirdai::ar

// Unregistered type error without this.
// https://uscilab.github.io/cereal/assets/doxygen/polymorphic_8hpp.html#a8e0d5df9830c0ed7c60451cf2f873ff5
CEREAL_FORCE_DYNAMIC_INIT(ArchiveMap)  // NOLINT