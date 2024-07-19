#pragma once

#include <archive/src/Archive.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::ar {

/**
 * This is for storing concrete C++ types that contain the data needed to
 * deserialize an object. This class should only be used to store C++ types, not
 * any custom classes or structs. This is to ensure that it can always be
 * loaded, since the definition of C++ classes and types are stable. For example
 * if we added our own custom class here, then changed the fields in it,
 * serialization would break.
 */
template <typename T>
class Value final : public Archive {
 public:
  explicit Value(T value) : _value(std::move(value)) {}

  static std::shared_ptr<Value<T>> make(T value) {
    return std::make_shared<Value<T>>(std::move(value));
  }

  const T& value() const { return _value; }

  std::string type() const final { return typeName(); }

  static std::string typeName();

 private:
  T _value;

  Value() {}

  friend class cereal::access;

  template <class Ar>
  void save(Ar& archive) const;

  template <class Ar>
  void load(Ar& archive);
};

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define APPLY_TO_TYPES(EXPR)                                   \
  EXPR(bool)                                                   \
  EXPR(uint64_t)                                               \
  EXPR(int64_t)                                                \
  EXPR(float)                                                  \
  EXPR(char)                                                   \
  EXPR(std::string)                                            \
  EXPR(std::vector<uint32_t>)                                  \
  EXPR(std::vector<float>)                                     \
  EXPR(std::vector<uint64_t>)                                  \
  EXPR(std::vector<int64_t>)                                   \
  EXPR(std::vector<std::string>)                               \
  EXPR(std::vector<std::wstring>)                              \
  EXPR(std::vector<std::vector<uint32_t>>)                     \
  EXPR(std::vector<std::vector<float>>)                        \
  EXPR(std::unordered_map<uint64_t, uint64_t>)                 \
  EXPR(std::unordered_map<uint64_t, std::vector<uint64_t>>)    \
  EXPR(std::unordered_map<uint64_t, std::vector<float>>)       \
  EXPR(std::unordered_map<std::string, uint64_t>)              \
  EXPR(std::unordered_map<std::string, int64_t>)               \
  EXPR(std::unordered_map<std::string, uint32_t>)              \
  EXPR(std::unordered_map<std::string, std::vector<uint64_t>>) \
  EXPR(std::unordered_map<int64_t, std::vector<std::string>>)

}  // namespace thirdai::ar