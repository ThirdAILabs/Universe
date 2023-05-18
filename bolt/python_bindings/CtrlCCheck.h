#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

namespace thirdai::bolt::train::python {

struct CtrlCCheck {
  void operator()() {
    if (PyErr_CheckSignals() != 0) {
      throw py::error_already_set();
    }
  }
};

}  // namespace thirdai::bolt::train::python