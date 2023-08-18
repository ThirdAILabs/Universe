#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

struct CtrlCCheck {
  void operator()() {
    // Reference:
    // https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-properly-handle-ctrl-c-in-long-running-functions
    if (PyErr_CheckSignals() != 0) {
      throw py::error_already_set();
    }
  }
};

}  // namespace thirdai::bolt::python