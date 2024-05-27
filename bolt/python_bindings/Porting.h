#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

py::dict modelParams(const ModelPtr& model);

ModelPtr modelFromParams(const py::dict& params);

}  // namespace thirdai::bolt::python