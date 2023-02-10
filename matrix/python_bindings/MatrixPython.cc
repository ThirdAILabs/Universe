#include "MatrixPython.h"
#include <matrix/src/EigenOps.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace thirdai::matrix::python {

void createMatrixSubmodule(py::module_& module) {
  auto matrix_submodule = module.def_submodule("matrix");

  matrix_submodule.def("eigen_mult", &eigenMult, py::arg("x"), py::arg("y"));
}

}  // namespace thirdai::matrix::python