#pragma once

#include <hashing/src/DensifiedMinHash.h>
#include <hashing/src/FastSRP.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace thirdai::hashing::python {

void createHashingSubmodule(py::module_& module);

}  // namespace thirdai::hashing::python
