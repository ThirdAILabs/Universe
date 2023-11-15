#include "Value.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <archive/src/StringCipher.h>
#include <unordered_map>

namespace thirdai::ar {

template <>
std::string Value<bool>::typeName() {
  return "Value[bool]";
}

template <>
std::string Value<uint64_t>::typeName() {
  return "Value[uint64_t]";
}

template <>
std::string Value<int64_t>::typeName() {
  return "Value[int64_t]";
}

template <>
std::string Value<float>::typeName() {
  return "Value[float]";
}

template <>
std::string Value<std::string>::typeName() {
  return "Value[std::string]";
}

template <>
std::string Value<std::vector<uint32_t>>::typeName() {
  return "Value[std::vector<uint32_t>]";
}

template <>
std::string Value<std::vector<int64_t>>::typeName() {
  return "Value[std::vector<int64_t>]";
}

template <>
std::string Value<std::vector<std::string>>::typeName() {
  return "Value[std::vector<std::string>]";
}

template <>
std::string Value<std::vector<std::wstring>>::typeName() {
  return "Value[std::vector<std::wstring>]";
}

template <>
std::string
Value<std::unordered_map<uint64_t, std::vector<uint64_t>>>::typeName() {
  return "Value[std::unordered_map<uint64_t, std::vector<uint64_t>>]";
}

template <>
std::string
Value<std::unordered_map<uint64_t, std::vector<float>>>::typeName() {
  return "Value[std::unordered_map<uint64_t, std::vector<float>>]";
}

template <>
template <class Ar>
void Value<std::string>::save(Ar& archive) const {
  std::string cipher_value = cipher(_value);
  archive(cereal::base_class<Archive>(this), cipher_value);
}

template <typename T>
template <class Ar>
void Value<T>::save(Ar& archive) const {
  archive(cereal::base_class<Archive>(this), _value);
}

template <>
template <class Ar>
void Value<std::string>::load(Ar& archive) {
  std::string cipher_value;
  archive(cereal::base_class<Archive>(this), cipher_value);
  _value = cipher(cipher_value);
}

template <typename T>
template <class Ar>
void Value<T>::load(Ar& archive) {
  archive(cereal::base_class<Archive>(this), _value);
}

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define SPECIALIZE_ARCHIVE_VALUE_SAVE(...) \
  template void Value<__VA_ARGS__>::save(cereal::BinaryOutputArchive&) const;

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define SPECIALIZE_ARCHIVE_VALUE_LOAD(...) \
  template void Value<__VA_ARGS__>::load(cereal::BinaryOutputArchive&);

APPLY_TO_TYPES(SPECIALIZE_ARCHIVE_VALUE_SAVE)
APPLY_TO_TYPES(SPECIALIZE_ARCHIVE_VALUE_LOAD)

}  // namespace thirdai::ar

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define REGISTER_ARCHIVE_VALUE_TYPE(...) \
  CEREAL_REGISTER_TYPE(thirdai::ar::Value<__VA_ARGS__>)

APPLY_TO_TYPES(REGISTER_ARCHIVE_VALUE_TYPE)