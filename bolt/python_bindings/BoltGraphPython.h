#pragma once

#include <pybind11/pybind11.h>
#include <bolt/src/graph/Graph.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

void createBoltGraphSubmodule(py::module_& bolt_submodule);

template<typename MODEL_T>
py::tuple predictPythonWrapper(MODEL_T& model, const py::object& data, const py::object& labels,
             const PredictConfig& predict_config);

}  // namespace thirdai::bolt::python
