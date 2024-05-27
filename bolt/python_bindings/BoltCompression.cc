#include "BoltCompression.h"
#include "PybindUtils.h"
#include <compression/python_bindings/ConversionUtils.h>
#include <compression/src/CompressionFactory.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <utils/Random.h>
#include <optional>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::bolt::compression::python {

template <typename T>
using NumpyArray = thirdai::bolt::python::NumpyArray<T>;

using SerializedCompressedVector =
    py::array_t<char, py::array::c_style | py::array::forcecast>;

using FloatCompressedVector =
    std::variant<thirdai::compression::DragonVector<float>,
                 thirdai::compression::CountSketch<float>>;

template <typename T>
SerializedCompressedVector compress(const py::array_t<T>& data,
                                    const std::string& compression_scheme,
                                    float compression_density,
                                    uint32_t seed_for_hashing,
                                    uint32_t sample_population_size) {
  FloatCompressedVector compressed_vector = thirdai::compression::compress(
      data.data(), static_cast<uint32_t>(data.size()), compression_scheme,
      compression_density, seed_for_hashing, sample_population_size);

  uint32_t serialized_size =
      std::visit(thirdai::compression::SizeVisitor<float>(), compressed_vector);

  char* serialized_compressed_vector = new char[serialized_size];

  std::visit(thirdai::compression::SerializeVisitor<float>(
                 serialized_compressed_vector),
             compressed_vector);

  py::capsule free_when_done(serialized_compressed_vector,
                             [](void* ptr) { delete static_cast<char*>(ptr); });

  return SerializedCompressedVector(
      serialized_size, serialized_compressed_vector, free_when_done);
}

template <typename T>
py::array_t<T> decompress(SerializedCompressedVector& compressed_vector) {
  const char* serialized_data =
      py::cast<SerializedCompressedVector>(compressed_vector).data();
  FloatCompressedVector des_compressed_vector =
      thirdai::compression::python::deserializeCompressedVector<float>(
          serialized_data);
  std::vector<float> full_gradients = std::visit(
      thirdai::compression::DecompressVisitor<float>(), des_compressed_vector);

  return py::array_t<T>(full_gradients.size(), full_gradients.data());
}

template <typename T>
py::array_t<T> concat(const py::object& compressed_vectors) {
  std::vector<FloatCompressedVector> py_compressed_vectors =
      thirdai::compression::python::convertPyListToCompressedVectors<float>(
          compressed_vectors);
  FloatCompressedVector concatenated_compressed_vector =
      thirdai::compression::concat(std::move(py_compressed_vectors));

  uint32_t serialized_size =
      std::visit(thirdai::compression::SizeVisitor<float>(),
                 concatenated_compressed_vector);

  char* serialized_compressed_vector = new char[serialized_size];

  std::visit(thirdai::compression::SerializeVisitor<float>(
                 serialized_compressed_vector),
             concatenated_compressed_vector);

  py::capsule free_when_done(serialized_compressed_vector,
                             [](void* ptr) { delete static_cast<char*>(ptr); });

  return SerializedCompressedVector(
      serialized_size, serialized_compressed_vector, free_when_done);
}

void createCompressionSubmodule(py::module_& module) {
  auto compression =
      module
          .def_submodule("compression")

          .def(
              "compress", &compress<float>, py::arg("data"),
              py::arg("compression_scheme"), py::arg("compression_density"),
              py::arg("seed_for_hashing"), py::arg("sample_population_size"),
              "Returns a char array representing a compressed vector. "
              "sample_population_size is the number of random samples you take "
              "for estimating a threshold for dragon compression or the number "
              "of sketches needed for count_sketch")
          .def("decompress", &decompress<float>, py::arg("compressed_vector"))
          .def("concat", &concat<float>, py::arg("compressed_vectors"));
}

}  // namespace thirdai::bolt::compression::python
