#pragma once

#include <serialization/src/Archive.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace thirdai::serialization {

template <typename T>
class ArchiveValue final : public Archive {
 public:
  explicit ArchiveValue(T value) : _value(std::move(value)) {}

  static std::shared_ptr<ArchiveValue<T>> make(T value) {
    return std::make_shared<ArchiveValue<T>>(std::move(value));
  }

  const T& get() const { return _value; }

 private:
  T _value;
};

}  // namespace thirdai::serialization