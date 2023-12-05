#pragma once

#include <stdexcept>
#include <string>

namespace thirdai::smx {

enum Dtype { u32, f32 };

inline size_t sizeofDtype(Dtype dtype) {
  switch (dtype) {
    case Dtype::f32:
      return sizeof(float);
    case Dtype::u32:
      return sizeof(uint32_t);
    default:
      throw std::invalid_argument("Encountered invalid dtype.");
  }
}

inline std::string toString(Dtype dtype) {
  switch (dtype) {
    case Dtype::f32:
      return "f32";
    case Dtype::u32:
      return "u32";
    default:
      throw std::invalid_argument("Encountered invalid dtype.");
  }
}

template <typename T>
inline Dtype getDtype();

template <>
inline Dtype getDtype<uint32_t>() {
  return Dtype::u32;
}

template <>
inline Dtype getDtype<float>() {
  return Dtype::f32;
}

template <typename T>
inline Dtype getDtype() {
  throw std::invalid_argument("Unsupported dtype.");
}

}  // namespace thirdai::smx