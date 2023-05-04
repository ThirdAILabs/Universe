#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace thirdai::automl::udt {

namespace py = pybind11;

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

}  // namespace thirdai::automl::udt
