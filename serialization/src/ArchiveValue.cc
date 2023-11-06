#include "ArchiveValue.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <unordered_map>

namespace thirdai::serialization {

template <>
std::string ArchiveValue<bool>::typeName() {
  return "Value[bool]";
}

template <>
std::string ArchiveValue<uint64_t>::typeName() {
  return "Value[uint64_t]";
}

template <>
std::string ArchiveValue<int64_t>::typeName() {
  return "Value[int64_t]";
}

template <>
std::string ArchiveValue<float>::typeName() {
  return "Value[float]";
}

template <>
std::string ArchiveValue<std::string>::typeName() {
  return "Value[std::string]";
}

template <>
std::string ArchiveValue<std::vector<uint32_t>>::typeName() {
  return "Value[std::vector<uint32_t>]";
}

template <>
std::string ArchiveValue<std::vector<std::string>>::typeName() {
  return "Value[std::vector<std::string>]";
}

template <typename T>
template <class Ar>
void ArchiveValue<T>::save(Ar& archive) const {
  archive(cereal::base_class<Archive>(this), _value);
}

template <typename T>
template <class Ar>
void ArchiveValue<T>::load(Ar& archive) {
  archive(cereal::base_class<Archive>(this), _value);
}

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define SPECIALIZE_ARCHIVE_VALUE_SAVE(...)                                    \
  template void ArchiveValue<__VA_ARGS__>::save(cereal::BinaryOutputArchive&) \
      const;

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define SPECIALIZE_ARCHIVE_VALUE_LOAD(...) \
  template void ArchiveValue<__VA_ARGS__>::load(cereal::BinaryOutputArchive&);

APPLY_TO_TYPES(SPECIALIZE_ARCHIVE_VALUE_SAVE)
APPLY_TO_TYPES(SPECIALIZE_ARCHIVE_VALUE_LOAD)

}  // namespace thirdai::serialization

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define REGISTER_ARCHIVE_VALUE_TYPE(...) \
  CEREAL_REGISTER_TYPE(thirdai::serialization::ArchiveValue<__VA_ARGS__>)

APPLY_TO_TYPES(REGISTER_ARCHIVE_VALUE_TYPE)