#pragma once
#include <compression/src/CompressedVector.h>
#include <compression/src/DragonVector.h>
#include <cstdint>
#include <memory>
#include <stdexcept>

// To-Do: What headers do we actually need here?
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace thirdai::compression::python {

template <class T>
inline std::vector<T> make_vector_from_1d_numpy_array(
    const py::array_t<T>& py_array) {
  return std::vector<T>(py_array.data(), py_array.data() + py_array.size());
}

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

// To-Do(Shubh): We will have to remove this function and work directly with
// binary streams to serialize and deserialize objects
inline std::unique_ptr<CompressedVector<float>> convertPyDictToCompressedVector(
    const py::object& pycompressed_vector) {
  using thirdai::compression::python::NumpyArray;

  if (py::cast<std::string>(pycompressed_vector["compression_scheme"]) ==
      "dragon") {
    NumpyArray<uint32_t> indices =
        pycompressed_vector["indices"].cast<NumpyArray<uint32_t>>();
    NumpyArray<float> values =
        pycompressed_vector["values"].cast<NumpyArray<float>>();

    std::vector<uint32_t> vector_indices =
        thirdai::compression::python::make_vector_from_1d_numpy_array(
            py::cast<py::array_t<uint32_t>>(indices));

    std::vector<float> vector_values =
        thirdai::compression::python::make_vector_from_1d_numpy_array(
            py::cast<py::array_t<float>>(values));
    DragonVector<float> dragon_sketch = compression::DragonVector<float>(
        std::move(vector_indices), std::move(vector_values),
        py::cast<std::uint32_t>(pycompressed_vector["original_size"]),
        py::cast<int>(pycompressed_vector["seed_for_hashing"]));

    return std::make_unique<DragonVector<float>>(dragon_sketch);
  }
  throw std::logic_error("Only dragon vectors are supported as of now");
}

inline py::dict convertCompressedVectorToPyDict(
    const std::unique_ptr<CompressedVector<float>>& compressed_vector) {
  py::dict py_compressed_vector;

  py_compressed_vector["compression_scheme"] =
      compressed_vector->getCompressionScheme();
  if (compressed_vector->getCompressionScheme() == "dragon") {
    // dynamic casting a compressed vector to a dragon vector
    DragonVector<float> dragon_sketch =
        *dynamic_cast<DragonVector<float>*>(compressed_vector.get());

    py_compressed_vector["original_size"] = dragon_sketch.getOriginalSize();
    py_compressed_vector["sketch_size"] = dragon_sketch.size();
    py_compressed_vector["seed_for_hashing"] =
        dragon_sketch.getSeedForHashing();
    py_compressed_vector["compression_density"] =
        dragon_sketch.getCompressionDensity();
    py_compressed_vector["indices"] =
        py::array_t<uint32_t>(py::cast(dragon_sketch.getIndices()));
    py_compressed_vector["values"] =
        py::array_t<float>(py::cast(dragon_sketch.getValues()));

    return py_compressed_vector;
  }

  throw std::logic_error("CompressedVector Type not known");
  return py_compressed_vector;
}

}  // namespace thirdai::compression::python