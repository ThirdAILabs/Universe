#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::automl::python {

void addPretrainedBaseModule(py::module_& module);

}  // namespace thirdai::automl::python