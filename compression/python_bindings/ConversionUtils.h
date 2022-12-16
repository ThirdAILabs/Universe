#pragma once

#include <compression/src/CompressedVector.h>
#include <compression/src/CompressionFactory.h>
#include <compression/src/CountSketch.h>
#include <compression/src/DragonVector.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <variant>
namespace py = pybind11;

namespace thirdai::compression::python {
using SerializedCompressedVector =
    py::array_t<char, py::array::c_style | py::array::forcecast>;

template <class T>
using CompressedVector = std::variant<DragonVector<T>, CountSketch<T>>;

template <class T>
std::variant<DragonVector<T>, CountSketch<T>> deserializeCompressedVector(
    const char* serialized_compressed_vector) {
  int compression_scheme;
  std::memcpy(&compression_scheme, serialized_compressed_vector,
              sizeof(uint32_t));

  CompressionScheme compression_scheme_enum =
      static_cast<CompressionScheme>(compression_scheme);

  // std::variant automatically binds
  switch (compression_scheme_enum) {
    case CompressionScheme::Dragon:
      return DragonVector<T>(serialized_compressed_vector);
    case CompressionScheme::CountSketch:
      return CountSketch<T>(serialized_compressed_vector);
  }

  throw std::invalid_argument(
      "Compression Scheme not supported. Only supports dragon and "
      "count_sketch");
}

template <class T>
std::vector<std::variant<DragonVector<T>, CountSketch<T>>>
convertPyListToCompressedVectors(const py::list& py_compressed_vectors) {
  std::vector<std::variant<DragonVector<T>, CountSketch<T>>> compressed_vectors;
  size_t num_vectors = py_compressed_vectors.size();
  compressed_vectors.reserve(num_vectors);

  for (size_t i = 0; i < num_vectors; i++) {
    const char* serialized_data =
        py::cast<SerializedCompressedVector>(py_compressed_vectors[i]).data();
    compressed_vectors.emplace_back(
        deserializeCompressedVector<T>(serialized_data));
  }
  return compressed_vectors;
}

template <class T>
SerializedCompressedVector createNumpyArrayFromCompressedVector(
    uint32_t serialized_size, char* serialized_compressed_vector,
    CompressedVector<T> compressed_vector) {
  std::visit(
      thirdai::compression::SerializeVisitor<T>(serialized_compressed_vector),
      compressed_vector);

  py::capsule free_when_done(serialized_compressed_vector,
                             [](void* ptr) { delete static_cast<char*>(ptr); });

  return SerializedCompressedVector(
      serialized_size, serialized_compressed_vector, free_when_done);
}
}  // namespace thirdai::compression::python