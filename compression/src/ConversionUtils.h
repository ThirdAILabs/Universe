#pragma once

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::compression::python {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

inline bool isNumpyArray(const py::object& obj) {
  return py::str(obj.get_type()).equal(py::str("<class 'numpy.ndarray'>"));
}

inline py::str getDtype(const py::object& obj) {
  return py::str(obj.attr("dtype"));
}

inline bool checkNumpyDtype(const py::object& obj, const std::string& type) {
  return getDtype(obj).equal(py::str(type));
}

inline bool checkNumpyDtypeFloat32(const py::object& obj) {
  return checkNumpyDtype(obj, "float32");
}

inline bool checkNumpyDtypeInt32(const py::object& obj) {
  return checkNumpyDtype(obj, "int32");
}

inline bool checkNumpyDtypeAnyInt(const py::object& obj) {
  return checkNumpyDtype(obj, "int32") || checkNumpyDtype(obj, "uint32") ||
         checkNumpyDtype(obj, "int64") || checkNumpyDtype(obj, "uint64");
}

inline bool checkNumpyDtypeUint32(const py::object& obj) {
  return checkNumpyDtype(obj, "uint32");
}

// // // template <class T>
// inline std::vector<uint32_t> make_vector_from_1d_numpy_array(
//     const py::array_t<uint32_t>& py_array) {
//   return std::vector<uint32_t>(py_array.data(),
//                                py_array.data() + py_array.size());
// }

// inline std::vector<float> make_vector_from_1d_numpy_array(
//     const py::array_t<float>& py_array) {
//   return std::vector<float>(py_array.data(), py_array.data() +
//   py_array.size());
// }

template <class T>
inline std::vector<T> make_vector_from_1d_numpy_array(
    const py::array_t<T>& py_array) {
  return std::vector<T>(py_array.data(), py_array.data() + py_array.size());
}

}  // namespace thirdai::compression::python
