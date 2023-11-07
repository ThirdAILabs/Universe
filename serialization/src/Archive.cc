#include "Archive.h"
#include <cereal/archives/binary.hpp>
#include <serialization/src/ArchiveList.h>
#include <serialization/src/ArchiveMap.h>
#include <serialization/src/ArchiveValue.h>
#include <serialization/src/ParameterReference.h>
#include <stdexcept>

namespace thirdai::ar {

const ArchiveMap& Archive::map() const {
  const auto* map = dynamic_cast<const ArchiveMap*>(this);
  if (!map) {
    throw std::invalid_argument(
        "Expected to the archive to have type Map but found '" + type() + "'.");
  }
  return *map;
}

const ArchiveList& Archive::list() const {
  const auto* list = dynamic_cast<const ArchiveList*>(this);
  if (!list) {
    throw std::invalid_argument(
        "Expected to the archive to have type List but found '" + type() +
        "'.");
  }
  return *list;
}

const ParameterReference& Archive::param() const {
  const auto& param = dynamic_cast<const ParameterReference*>(this);
  if (!param) {
    throw std::invalid_argument(
        "Expected to the archive to have type ParameterReference but found '" +
        type() + "'.");
  }
  return *param;
}

bool Archive::contains(const std::string& key) const {
  return map().contains(key);
}

const ConstArchivePtr& Archive::at(const std::string& key) const {
  return map().get(key);
}

template <typename T>
const T& Archive::get() const {
  const auto* val = dynamic_cast<const ArchiveValue<T>*>(this);
  if (!val) {
    throw std::invalid_argument("Expected to the archive to have type '" +
                                ArchiveValue<T>::typeName() + "' but found '" +
                                type() + "'.");
  }
  return val->get();
}

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define SPECIALIZE_get(...) template const __VA_ARGS__& Archive::get() const;

APPLY_TO_TYPES(SPECIALIZE_get)

template <typename T>
bool Archive::is() const {
  return dynamic_cast<const ArchiveValue<T>*>(this);
}

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define SPECIALIZE_is(...) template bool Archive::is<__VA_ARGS__>() const;

APPLY_TO_TYPES(SPECIALIZE_is)

template <typename T>
const T& Archive::get(const std::string& key) const {
  return at(key)->get<T>();
}

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define SPECIALIZE_get_key(...) \
  template const __VA_ARGS__& Archive::get(const std::string&) const;

APPLY_TO_TYPES(SPECIALIZE_get_key)

template <typename T>
const T& Archive::getOr(const std::string& key, const T& fallback) const {
  if (contains(key)) {
    return get<T>(key);
  }
  return fallback;
}

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define SPECIALIZE_getOr(...)                                    \
  template const __VA_ARGS__& Archive::getOr(const std::string&, \
                                             const __VA_ARGS__&) const;

APPLY_TO_TYPES(SPECIALIZE_getOr)

template <typename T>
std::optional<T> Archive::getOpt(const std::string& key) const {
  if (contains(key)) {
    return get<T>(key);
  }
  return std::nullopt;
}

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define SPECIALIZE_getOpt(...) \
  template std::optional<__VA_ARGS__> Archive::getOpt(const std::string&) const;

APPLY_TO_TYPES(SPECIALIZE_getOpt)

template void Archive::save(cereal::BinaryOutputArchive&) const;

template <class Ar>
void Archive::save(Ar& archive) const {
  (void)archive;
}

template void Archive::load(cereal::BinaryInputArchive&);

template <class Ar>
void Archive::load(Ar& archive) {
  (void)archive;
}

ConstArchivePtr boolean(bool val) { return ArchiveValue<bool>::make(val); }

ConstArchivePtr u64(uint64_t val) { return ArchiveValue<uint64_t>::make(val); }

ConstArchivePtr i64(int64_t val) { return ArchiveValue<int64_t>::make(val); }

ConstArchivePtr f32(float val) { return ArchiveValue<float>::make(val); }

ConstArchivePtr str(std::string val) {
  return ArchiveValue<std::string>::make(std::move(val));
}

ConstArchivePtr vec(std::vector<uint32_t> val) {
  return ArchiveValue<std::vector<uint32_t>>::make(std::move(val));
}

ConstArchivePtr vec(std::vector<int64_t> val) {
  return ArchiveValue<std::vector<int64_t>>::make(std::move(val));
}

ConstArchivePtr vec(std::vector<std::string> val) {
  return ArchiveValue<std::vector<std::string>>::make(std::move(val));
}

ConstArchivePtr vec(std::vector<std::wstring> val) {
  return ArchiveValue<std::vector<std::wstring>>::make(std::move(val));
}

ConstArchivePtr map(std::unordered_map<uint64_t, std::vector<uint64_t>> val) {
  return ArchiveValue<std::unordered_map<uint64_t, std::vector<uint64_t>>>::
      make(std::move(val));
}

ConstArchivePtr map(std::unordered_map<uint64_t, std::vector<float>> val) {
  return ArchiveValue<std::unordered_map<uint64_t, std::vector<float>>>::make(
      std::move(val));
}

}  // namespace thirdai::ar