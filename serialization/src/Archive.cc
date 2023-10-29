#include "Archive.h"
#include <serialization/src/ArchiveList.h>
#include <serialization/src/ArchiveMap.h>
#include <serialization/src/ArchiveValue.h>
#include <stdexcept>

namespace thirdai::serialization {

const ArchiveMap& Archive::map() const {
  const auto* map = dynamic_cast<const ArchiveMap*>(this);
  if (!map) {
    throw std::invalid_argument("The archive does not have type map.");
  }
  return *map;
}

const ArchiveList& Archive::list() const {
  const auto* list = dynamic_cast<const ArchiveList*>(this);
  if (!list) {
    throw std::invalid_argument("The archive does not have type list.");
  }
  return *list;
}

bool Archive::contains(const std::string& key) const {
  return map().contains(key);
}

const ConstArchivePtr& Archive::at(const std::string& key) const {
  return map().at(key);
}

template <typename T>
const T& Archive::get() const {
  const auto* value = dynamic_cast<const ArchiveValue<T>*>(this);
  if (value) {
    return value->get();
  }
  throw std::invalid_argument(
      "Archive does not contain primitive of the requested type.");
}

template <typename T>
const T& Archive::get(const std::string& key) const {
  return at(key)->get<T>();
}

template <typename T>
const T& Archive::getOr(const std::string& key, const T& fallback) const {
  if (contains(key)) {
    return get<T>(key);
  }
  return fallback;
}

}  // namespace thirdai::serialization