#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::automl {

std::optional<float> floatArg(const py::kwargs& kwargs, const std::string& key);

}  // namespace thirdai::automl