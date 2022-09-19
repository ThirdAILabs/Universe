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

namespace py = pybind11;

namespace thirdai::compression::python {

template <class T>
inline std::vector<T> makeVectorFrom1dNumpyArray(
    const py::array_t<T>& py_array);

// TODO(Shubh): Profiling this function to see if copying is a bottleneck.
template <class T>
inline std::vector<T> makeVectorFrom1dNumpyArray(
    const py::array_t<T>& py_array) {
  return std::vector<T>(py_array.data(), py_array.data() + py_array.size());
}

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

using SerializedCompressedVector =
    py::array_t<char, py::array::c_style | py::array::forcecast>;

inline std::unique_ptr<CompressedVector<float>> deserializeCompressedVector(
    const char* serialized_compressed_vector) {
  /*
   * We find the compression scheme from the char array and then pass the char
   * array to the constructor of the respective class.
   */
  uint32_t compression_scheme;
  std::memcpy(reinterpret_cast<char*>(&compression_scheme),
              serialized_compressed_vector, sizeof(uint32_t));

  CompressionScheme compression_scheme_enum =
      static_cast<CompressionScheme>(compression_scheme);

  switch (compression_scheme_enum) {
    case CompressionScheme::Dragon: {
      DragonVector<float> compressed_vector(serialized_compressed_vector);
      return std::make_unique<DragonVector<float>>(
          std::move(compressed_vector));
    }
    case CompressionScheme::CountSketch: {
      CountSketch<float> compressed_vector(serialized_compressed_vector);
      return std::make_unique<CountSketch<float>>(std::move(compressed_vector));
    }
    default:
      throw std::logic_error(
          "Valid Compression Scheme could not be decoded from the serialized "
          "data. "
          "The serialized data has been corrupted.");
  }
}

inline void serializeCompressedVector(
    const std::unique_ptr<CompressedVector<float>>& compressed_vector,
    char* serialized_data) {
  compressed_vector->serialize(serialized_data);
}

inline std::vector<std::unique_ptr<CompressedVector<float>>>
convertPyListToCompressedVectors(const py::list& py_compressed_vectors) {
  int num_vectors = py_compressed_vectors.size();
  std::vector<std::unique_ptr<CompressedVector<float>>> compressed_vectors;
  compressed_vectors.reserve(num_vectors);
  for (int i = 0; i < num_vectors; i++) {
    compressed_vectors.push_back(deserializeCompressedVector(
        py::cast<SerializedCompressedVector>(py_compressed_vectors[i]).data()));
  }
  return compressed_vectors;
}
}  // namespace thirdai::compression::python