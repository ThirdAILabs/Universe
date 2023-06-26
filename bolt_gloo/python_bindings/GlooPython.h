#pragma once

#include <auto_ml/src/config/ArgumentMap.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace thirdai::bolt_gloo::python {
    void defineGlooSubmodule(py::module_& module);
}  // namespace thirdai::bolt_gloo::python