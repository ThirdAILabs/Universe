#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::ner::python {

void createNERModule(py::module_& module);

}  // namespace thirdai::bolt::ner::python