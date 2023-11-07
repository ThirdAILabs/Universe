#pragma once

#include <serialization/src/Archive.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::ar {

template <typename T>
class ArchiveValue final : public Archive {
 public:
  explicit ArchiveValue(T value) : _value(std::move(value)) {}

  static std::shared_ptr<ArchiveValue<T>> make(T value) {
    return std::make_shared<ArchiveValue<T>>(std::move(value));
  }

  const T& get() const { return _value; }

  std::string type() const final { return typeName(); }

  static std::string typeName();

 private:
  T _value;

  ArchiveValue() {}

  friend class cereal::access;

  template <class Ar>
  void save(Ar& archive) const;

  template <class Ar>
  void load(Ar& archive);
};

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define APPLY_TO_TYPES(EXPR)                                \
  EXPR(bool)                                                \
  EXPR(uint64_t)                                            \
  EXPR(int64_t)                                             \
  EXPR(float)                                               \
  EXPR(std::string)                                         \
  EXPR(std::vector<uint32_t>)                               \
  EXPR(std::vector<int64_t>)                                \
  EXPR(std::vector<std::string>)                            \
  EXPR(std::vector<std::wstring>)                           \
  EXPR(std::unordered_map<uint64_t, std::vector<uint64_t>>) \
  EXPR(std::unordered_map<uint64_t, std::vector<float>>)

}  // namespace thirdai::ar