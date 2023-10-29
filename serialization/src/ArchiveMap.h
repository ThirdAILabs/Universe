#pragma once

#include <serialization/src/Archive.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::serialization {

class ArchiveMap final : public Archive {
 public:
  static std::shared_ptr<ArchiveMap> make() {
    return std::make_shared<ArchiveMap>();
  }

  bool contains(const std::string& key) const { return _map.count(key); }

  const ConstArchivePtr& at(const std::string& key) const {
    if (contains(key)) {
      return _map.at(key);
    }
    throw std::invalid_argument("Archive contains no element for key '" + key +
                                "'.");
  }

  ConstArchivePtr& at(const std::string& key) { return _map[key]; }

  size_t size() const { return _map.size(); }

  auto begin() const { return _map.begin(); }

  auto end() const { return _map.end(); }

 private:
  std::unordered_map<std::string, ConstArchivePtr> _map;
};

}  // namespace thirdai::serialization