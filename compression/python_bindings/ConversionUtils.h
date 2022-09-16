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
#include <sstream>
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

inline std::unique_ptr<CompressedVector<float>> convertStringToCompressedVector(
    const std::string& compressed_vector) {
  if (compressed_vector[4] == 'd') {
    std::stringstream ss(compressed_vector);
    DragonVector<float> dragon_vector = DragonVector<float>(ss);
    return std::make_unique<DragonVector<float>>(dragon_vector);
  }
  throw std::logic_error("eff");
}

inline py::bytes convertCompressedVectorToString(
    const std::unique_ptr<CompressedVector<float>>& compressed_vector) {
  py::bytes bytes_compressed_vector(compressed_vector->serialize().str());
  // an implicit copy being made here because of conversion from string to bytes
  return bytes_compressed_vector;
}

inline std::vector<std::unique_ptr<CompressedVector<float>>>
convertPyListToCompressedVectors(const py::list& py_compressed_vectors) {
  int num_vectors = py_compressed_vectors.size();
  std::vector<std::unique_ptr<CompressedVector<float>>> compressed_vectors;
  compressed_vectors.reserve(num_vectors);
  for (int i = 0; i < num_vectors; i++) {
    compressed_vectors.push_back(convertStringToCompressedVector(
        py::cast<py::bytes>(py_compressed_vectors[i])));
  }
  return compressed_vectors;
}
}  // namespace thirdai::compression::python