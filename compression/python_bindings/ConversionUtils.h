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

inline std::unique_ptr<CompressedVector<float>> convertPyDictToCompressedVector(
    const py::object& pycompressed_vector);

inline py::dict convertCountSketchToPyDict(
    const CountSketch<float>& count_sketch_vector);

inline py::dict convertDragonVectorToPyDict(
    const DragonVector<float>& dragon_sketch);

inline std::unique_ptr<CompressedVector<float>> convertPydictToDragonVector(
    const py::object& pycompressed_vector);

inline std::unique_ptr<CompressedVector<float>> convertPyDictToCountSketch(
    const py::object& pycompressed_vector);

// TODO(Shubh): Profiling this function to see if copying is a bottleneck.
template <class T>
inline std::vector<T> makeVectorFrom1dNumpyArray(
    const py::array_t<T>& py_array) {
  return std::vector<T>(py_array.data(), py_array.data() + py_array.size());
}

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

// TODO(Shubh): We will have to remove this function and work directly with
// binary streams to serialize and deserialize objects
inline std::unique_ptr<CompressedVector<float>> convertPyDictToCompressedVector(
    const py::object& pycompressed_vector) {
  using thirdai::compression::python::NumpyArray;

  std::string compression_scheme =
      py::cast<std::string>(pycompressed_vector["compression_scheme"]);

  if (compression_scheme == "dragon") {
    return convertPydictToDragonVector(pycompressed_vector);
  }
  if (compression_scheme == "count_sketch") {
    return convertPyDictToCountSketch(pycompressed_vector);
  }
  throw std::logic_error(
      "Received unknown compression type " +
      py::cast<std::string>(pycompressed_vector["compression_scheme"]) +
      ". Currently only Dragon compression is supported.");
}

inline py::dict convertCompressedVectorToPyDict(
    const std::unique_ptr<CompressedVector<float>>& compressed_vector) {
  py::dict pycompressed_vector;
  pycompressed_vector["compression_scheme"] = compressed_vector->type();
  if (compressed_vector->type() == "dragon") {
    // dynamic casting a compressed vector to a dragon vector
    DragonVector<float> dragon_sketch =
        *dynamic_cast<DragonVector<float>*>(compressed_vector.get());
    return convertDragonVectorToPyDict(dragon_sketch);
  }
  if (compressed_vector->type() == "count_sketch") {
    CountSketch<float> count_sketch =
        *dynamic_cast<CountSketch<float>*>(compressed_vector.get());
    return convertCountSketchToPyDict(count_sketch);
  }
  throw std::logic_error("CompressedVector Type not known");
  return pycompressed_vector;
}

inline std::unique_ptr<CompressedVector<float>> convertPydictToDragonVector(
    const py::object& pycompressed_vector) {
  NumpyArray<uint32_t> indices =
      pycompressed_vector["indices"].cast<NumpyArray<uint32_t>>();

  NumpyArray<float> values =
      pycompressed_vector["values"].cast<NumpyArray<float>>();

  std::vector<uint32_t> vector_indices =
      thirdai::compression::python::makeVectorFrom1dNumpyArray(
          py::cast<py::array_t<uint32_t>>(indices));

  std::vector<float> vector_values =
      thirdai::compression::python::makeVectorFrom1dNumpyArray(
          py::cast<py::array_t<float>>(values));

  DragonVector<float> dragon_sketch = compression::DragonVector<float>(
      std::move(vector_indices), std::move(vector_values),
      py::cast<std::uint32_t>(pycompressed_vector["original_size"]),
      py::cast<int>(pycompressed_vector["seed_for_hashing"]));

  return std::make_unique<DragonVector<float>>(dragon_sketch);
}

inline py::dict convertDragonVectorToPyDict(
    const DragonVector<float>& dragon_sketch) {
  py::dict pycompressed_vector;
  pycompressed_vector["compression_scheme"] = dragon_sketch.type();
  pycompressed_vector["original_size"] = dragon_sketch.uncompressedSize();
  pycompressed_vector["sketch_size"] = dragon_sketch.size();
  pycompressed_vector["seed_for_hashing"] = dragon_sketch.seedForHashing();
  pycompressed_vector["compression_density"] =
      dragon_sketch.compressionDensity();
  pycompressed_vector["indices"] =
      py::array_t<uint32_t>(py::cast(dragon_sketch.indices()));
  pycompressed_vector["values"] =
      py::array_t<float>(py::cast(dragon_sketch.values()));
  return pycompressed_vector;
}

inline std::unique_ptr<CompressedVector<float>> convertPyDictToCountSketch(
    const py::object& pycompressed_vector) {
  py::list py_count_sketches = pycompressed_vector["count_sketches"];
  py::list py_seed_for_hashing_indices =
      pycompressed_vector["seed_for_hashing_indices"];
  py::list py_seed_for_sign = pycompressed_vector["seed_for_sign"];

  std::vector<std::vector<float>> cpp_count_sketches;
  std::vector<uint32_t> cpp_seed_for_hashing_indices;
  std::vector<uint32_t> cpp_seed_for_sign;

  uint32_t num_sketches = static_cast<uint32_t>(py_count_sketches.size());
  for (uint32_t num_sketch = 0; num_sketch < num_sketches; num_sketch++) {
    cpp_count_sketches.push_back(
        thirdai::compression::python::makeVectorFrom1dNumpyArray(
            py::cast<py::array_t<float>>(py_count_sketches[num_sketch])));
    cpp_seed_for_hashing_indices.push_back(
        py::cast<uint32_t>(py_seed_for_hashing_indices[num_sketch]));
    cpp_seed_for_sign.push_back(
        py::cast<uint32_t>(py_seed_for_sign[num_sketch]));
  }
  // std::cout << "sketches being made and all" << std::endl;

  uint32_t uncompressed_size =
      py::cast<uint32_t>(pycompressed_vector["uncompressed_size"]);

  CountSketch<float> count_sketch = compression::CountSketch<float>(
      cpp_count_sketches, cpp_seed_for_hashing_indices, cpp_seed_for_sign,
      uncompressed_size);
  // std::cout << "done calling the constructor" << std::endl;

  return std::make_unique<CountSketch<float>>(count_sketch);
}

inline py::dict convertCountSketchToPyDict(
    const CountSketch<float>& count_sketch_vector) {
  py::list count_sketches;
  py::list seed_for_hashing_indices;
  py::list seed_for_sign;

  for (uint32_t num_sketch = 0; num_sketch < count_sketch_vector.numSketches();
       num_sketch++) {
    count_sketches.append(py::array_t<float>(
        py::cast(count_sketch_vector.sketches()[num_sketch])));
    seed_for_hashing_indices.append(
        count_sketch_vector.indexSeeds()[num_sketch]);
    seed_for_sign.append(count_sketch_vector.signSeeds()[num_sketch]);
  }

  py::dict pycompressed_vector;
  pycompressed_vector["compression_scheme"] = count_sketch_vector.type();
  pycompressed_vector["count_sketches"] = count_sketches;
  pycompressed_vector["seed_for_hashing_indices"] = seed_for_hashing_indices;
  pycompressed_vector["seed_for_sign"] = seed_for_sign;
  pycompressed_vector["uncompressed_size"] =
      count_sketch_vector.uncompressedSize();

  return pycompressed_vector;
}
}  // namespace thirdai::compression::python