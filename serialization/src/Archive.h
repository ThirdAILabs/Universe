#pragma once

#include <vector>

namespace thirdai::serialization {

class Archive;
using ArchivePtr = std::shared_ptr<Archive>;
using ConstArchivePtr = std::shared_ptr<const Archive>;

class ArchiveMap;

class ArchiveList;

class Archive {
 public:
  const ArchiveMap& map() const;

  const ArchiveList& list() const;

  bool contains(const std::string& key) const;

  const ConstArchivePtr& at(const std::string& key) const;

  template <typename T>
  const T& get() const;

  template <typename T>
  const T& get(const std::string& key) const;

  template <typename T>
  const T& getOr(const std::string& key, const T& fallback) const;

  virtual ~Archive() = default;
};

}  // namespace thirdai::serialization