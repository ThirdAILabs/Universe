#include "MatrixPython.h"
#include <matrix/src/EigenOps.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/stl.h>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorTraits.h>

namespace thirdai::matrix::python {

void createMatrixSubmodule(py::module_& module) {
  auto matrix_submodule = module.def_submodule("matrix");

  matrix_submodule.def("eigen_mult", &eigenMult, py::arg("x"), py::arg("y"));

  matrix_submodule.def("eigen_2dconv", &eigenConv, py::arg("image"),
                       py::arg("filters"));

  matrix_submodule.def(
      "eigen_2dconv_tf", &tfEigenConv, py::arg("image"), py::arg("filters"),
      py::arg("row_stride") = 1, py::arg("col_stride") = 1,
      py::arg("padding_type") = 2, py::arg("row_in_stride") = 1,
      py::arg("col_in_stride") = 1, py::arg("padding_top") = 0,
      py::arg("padding_bottom") = 0, py::arg("padding_left") = 0,
      py::arg("padding_right") = 0);
}

}  // namespace thirdai::matrix::python