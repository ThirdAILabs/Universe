#include "Archive.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <serialization/src/ArchiveList.h>
#include <serialization/src/ArchiveMap.h>
#include <serialization/src/ArchiveValue.h>
#include <serialization/src/ParameterReference.h>
#include <stdexcept>

namespace thirdai::ar {

const ArchiveMap& Archive::map() const {
  const auto* map = dynamic_cast<const ArchiveMap*>(this);
  if (!map) {
    throw std::runtime_error(
        "Expected to the archive to have type Map but found '" + type() + "'.");
  }
  return *map;
}

const ArchiveList& Archive::list() const {
  const auto* list = dynamic_cast<const ArchiveList*>(this);
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
  const auto* val = dynamic_cast<const ArchiveValue<T>*>(this);
  if (!val) {
    throw std::runtime_error("Attempted to convert archive of type '" + type() +
                             "' to type '" + ArchiveValue<T>::typeName() +
                             "'.");
  }
  return val->value();
}

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define SPECIALIZE_as(...) template const __VA_ARGS__& Archive::as() const;

APPLY_TO_TYPES(SPECIALIZE_as)

template <typename T>
bool Archive::is() const {
  return dynamic_cast<const ArchiveValue<T>*>(this);
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

// Saving the base class directly lead to the derived class's serialization
// method not being invoked. Using this wrapper solved the issue.
struct ArchiveWrapper {
  ConstArchivePtr _archive;

  template <class Ar>
  void serialize(Ar& archive) {
    archive(_archive);
  }
};

void serialize(ConstArchivePtr archive, std::ostream& output) {
  cereal::BinaryOutputArchive oarchive(output);

  ArchiveWrapper wrappper{std::move(archive)};
  oarchive(wrappper);
}

ConstArchivePtr deserialize(std::istream& input) {
  cereal::BinaryInputArchive iarchive(input);

  ArchiveWrapper wrapper;
  // ArchivePtr deserialize_into(new Archive());
  iarchive(wrapper);
  return wrapper._archive;
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