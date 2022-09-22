#pragma once

#include <compression/src/CompressedVector.h>
#include <compression/src/CountSketch.h>
#include <compression/src/DragonVector.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <variant>

namespace py = pybind11;

namespace thirdai::compression::python {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

using SerializedCompressedVector =
    py::array_t<char, py::array::c_style | py::array::forcecast>;

template <class T>
inline DragonVector<T> deserializeDragonVector(
    const char* serialized_dragon_vector) {
  return DragonVector<T>(serialized_dragon_vector);
}

template <class T>
inline CountSketch<T> deserializeCountSketch(
    const char* serialized_count_sketch) {
  return CountSketch<T>(serialized_count_sketch);
}

inline void serialize(
    const std::unique_ptr<CompressedVector<float>>& compressed_vector,
    char* serialized_data) {
  compressed_vector->serialize(serialized_data);
}

template <class T>
inline std::vector<DragonVector<T>> convertPyListToDragonVectors(
    const py::list& py_compressed_vectors) {
  size_t num_vectors = py_compressed_vectors.size();
  std::vector<DragonVector<T>> dragon_vectors;
  dragon_vectors.reserve(num_vectors);

  for (size_t i = 0; i < num_vectors; i++) {
    dragon_vectors.emplace_back(deserializeDragonVector<T>(
        py::cast<SerializedCompressedVector>(py_compressed_vectors[i]).data()));
  }
  return dragon_vectors;
}

template <class T>
inline std::vector<CountSketch<T>> convertPyListToCountSketches(
    const py::list& py_compressed_vectors) {
  size_t num_vectors = py_compressed_vectors.size();
  std::vector<CountSketch<T>> count_sketches;
  count_sketches.reserve(num_vectors);

  for (size_t i = 0; i < num_vectors; i++) {
    count_sketches.emplace_back(deserializeCountSketch<T>(
        py::cast<SerializedCompressedVector>(py_compressed_vectors[i]).data()));
  }
  return count_sketches;
}

template <class T>
std::variant<DragonVector<T>, CountSketch<T>> convertToCompressedVector(
    const char* serialized_compressed_vector,
    const std::string& compression_scheme) {
  CompressionScheme compression_scheme_enum =
      convertStringToEnum(compression_scheme);

  // std::variant automatically binds
  switch (compression_scheme_enum) {
    case CompressionScheme::Dragon:
      return DragonVector<T>(serialized_compressed_vector);
    case CompressionScheme::CountSketch:
      return CountSketch<T>(serialized_compressed_vector);
  }
}

template <class T>
std::vector<std::variant<DragonVector<T>, CountSketch<T>>>
convertPyListToCompressedVector(const py::list& py_compressed_vectors,
                                const std::string& compression_scheme) {
  std::vector<std::variant<DragonVector<T>, CountSketch<T>>> compressed_vectors;
  size_t num_vectors = py_compressed_vectors.size();
  compressed_vectors.reserve(num_vectors);

  for (size_t i = 0; i < num_vectors; i++) {
    const char* serialized_data =
        py::cast<SerializedCompressedVector>(py_compressed_vectors[i]).data();
    compressed_vectors.emplace_back(
        convertToCompressedVector<T>(serialized_data, compression_scheme));
  }
  return compressed_vectors;
}
}  // namespace thirdai::compression::python