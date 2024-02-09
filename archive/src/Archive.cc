#include "Archive.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <archive/src/ParameterReference.h>
#include <archive/src/Value.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::ar {

const Map& Archive::map() const {
  const auto* map = dynamic_cast<const Map*>(this);
  if (!map) {
    throw std::runtime_error(
        "Expected to the archive to have type Map but found '" + type() + "'.");
  }
  return *map;
}

const List& Archive::list() const {
  const auto* list = dynamic_cast<const List*>(this);
  if (!list) {
    throw std::runtime_error(
        "Expected to the archive to have type List but found '" + type() +
        "'.");
  }
  return *list;
}

const ParameterReference& Archive::param() const {
  const auto& param = dynamic_cast<const ParameterReference*>(this);
  if (!param) {
    throw std::runtime_error(
        "Expected to the archive to have type ParameterReference but found '" +
        type() + "'.");
  }
  return *param;
}

bool Archive::contains(const std::string& key) const {
  (void)key;
  throw std::runtime_error("'contains' can only be called on Map archives.");
}

const ConstArchivePtr& Archive::get(const std::string& key) const {
  (void)key;
  throw std::runtime_error("'get' can only be called on Map archives.");
}

template <typename T>
const T& Archive::as() const {
  const auto* val = dynamic_cast<const Value<T>*>(this);
  if (!val) {
    throw std::runtime_error("Attempted to convert archive of type '" + type() +
                             "' to type '" + Value<T>::typeName() + "'.");
  }
  return val->value();
}

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define SPECIALIZE_as(...) template const __VA_ARGS__& Archive::as() const;

APPLY_TO_TYPES(SPECIALIZE_as)

template <typename T>
bool Archive::is() const {
  return dynamic_cast<const Value<T>*>(this);
}

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define SPECIALIZE_is(...) template bool Archive::is<__VA_ARGS__>() const;

APPLY_TO_TYPES(SPECIALIZE_is)

template <typename T>
const T& Archive::getAs(const std::string& key) const {
  return get(key)->as<T>();
}

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define SPECIALIZE_getAs(...) \
  template const __VA_ARGS__& Archive::getAs(const std::string&) const;

APPLY_TO_TYPES(SPECIALIZE_getAs)

template <typename T>
const T& Archive::getOr(const std::string& key, const T& fallback) const {
  if (contains(key)) {
    return getAs<T>(key);
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
    return getAs<T>(key);
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

void serialize(ConstArchivePtr archive, std::ostream& output) {
  cereal::BinaryOutputArchive oarchive(output);
  oarchive(archive);
}

ConstArchivePtr deserialize(std::istream& input) {
  cereal::BinaryInputArchive iarchive(input);

  ArchivePtr archive;
  iarchive(archive);
  return archive;
}

ConstArchivePtr boolean(bool val) { return Value<bool>::make(val); }

ConstArchivePtr u64(uint64_t val) { return Value<uint64_t>::make(val); }

ConstArchivePtr i64(int64_t val) { return Value<int64_t>::make(val); }

ConstArchivePtr f32(float val) { return Value<float>::make(val); }

ConstArchivePtr character(char val) { return Value<char>::make(val); }

ConstArchivePtr str(std::string val) {
  return Value<std::string>::make(std::move(val));
}

ConstArchivePtr vecU32(std::vector<uint32_t> val) {
  return Value<std::vector<uint32_t>>::make(std::move(val));
}

ConstArchivePtr vecU64(std::vector<uint64_t> val) {
  return Value<std::vector<uint64_t>>::make(std::move(val));
}

ConstArchivePtr vecI64(std::vector<int64_t> val) {
  return Value<std::vector<int64_t>>::make(std::move(val));
}

ConstArchivePtr vecStr(std::vector<std::string> val) {
  return Value<std::vector<std::string>>::make(std::move(val));
}

ConstArchivePtr vecWStr(std::vector<std::wstring> val) {
  return Value<std::vector<std::wstring>>::make(std::move(val));
}

ConstArchivePtr vecVecU32(std::vector<std::vector<uint32_t>> val) {
  return Value<std::vector<std::vector<uint32_t>>>::make(std::move(val));
}

ConstArchivePtr vecVecF32(std::vector<std::vector<float>> val) {
  return Value<std::vector<std::vector<float>>>::make(std::move(val));
}

ConstArchivePtr mapU64VecU64(MapU64VecU64 val) {
  return Value<MapU64VecU64>::make(std::move(val));
}

ConstArchivePtr mapU64VecF32(MapU64VecF32 val) {
  return Value<MapU64VecF32>::make(std::move(val));
}

ConstArchivePtr mapStrU64(std::unordered_map<std::string, uint64_t> val) {
  return Value<std::unordered_map<std::string, uint64_t>>::make(std::move(val));
}

ConstArchivePtr mapStrI64(std::unordered_map<std::string, int64_t> val) {
  return Value<std::unordered_map<std::string, int64_t>>::make(std::move(val));
}

ConstArchivePtr mapI64VecStr(MapI64VecStr val) {
  return Value<MapI64VecStr>::make(std::move(val));
}

}  // namespace thirdai::ar