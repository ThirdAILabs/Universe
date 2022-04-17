// Code to create thirdai modules
#include <bolt/python_bindings/BoltPython.h>
#include <hashing/python_bindings/HashingPython.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <flash/python_bindings/FlashPython.h>

// Pybind11 library
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <wrappers/src/EigenDenseWrapper.h>
#include <pybind11/eigen.h>



#ifndef __clang__
#include <omp.h>
#endif

namespace thirdai::matrix::python {

  using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  RowMatrixXf naive_mult(const Eigen::Ref<const RowMatrixXf>& m1, const Eigen::Ref<const RowMatrixXf> & m2) {
    RowMatrixXf result = RowMatrixXf::Zero(m1.rows(), m2.cols());
    #pragma omp parallel for
    for (uint32_t row = 0; row < m1.rows(); row++) {
      for (uint32_t col = 0; col < m2.cols(); col++) {
        for (uint32_t i = 0; i < m1.cols(); i++) {
          result(row, col) += m1(row, i) * m2(i, col);
        }
      }
    }
    return result;
  }

  RowMatrixXf naive_eigen_mult(const Eigen::Ref<const RowMatrixXf>& m1, const Eigen::Ref<const RowMatrixXf> & m2) {
    return m1 * m2;
  }

  RowMatrixXf fast_eigen_mult(const Eigen::Ref<const RowMatrixXf>& m1, const Eigen::Ref<const RowMatrixXf> & m2, uint32_t num_slices) {

    uint32_t slice_size = m1.rows() / num_slices;
    std::vector<std::pair<uint32_t, uint32_t>> slices;
    for (uint32_t start = 0; start < m1.rows(); start += slice_size) {
      uint32_t end = std::min<uint32_t>(start + slice_size, m1.rows());
      slices.emplace_back(start, end);
    }

    RowMatrixXf result(m1.rows(), m2.cols());

    #pragma omp parallel for
    for (auto & slice : slices) {
      uint32_t start = slice.first;
      uint32_t end = slice.second;
      Eigen::Map<const RowMatrixXf> m1_slice(m1.row(start).data(), end - start, m1.cols()); 
      RowMatrixXf slice_result = m1_slice * m2;
      for (uint32_t row = start; row < end; row++) {
        for (uint32_t col = 0; col < m2.cols(); col++) {
          result(row, col) = slice_result(row - start, col);
        }
      }
    }

    return result;

  }

  void createMatrixSubmodule(py::module_& module) {
    auto matrix_submodule = module.def_submodule("matrix");
    matrix_submodule.def("naive_matmul", &naive_mult, py::arg("m1").noconvert(), py::arg("m2").noconvert());
    matrix_submodule.def("naive_eigen_matmul", &naive_eigen_mult, py::arg("m1").noconvert(), py::arg("m2").noconvert());
    matrix_submodule.def("fast_eigen_matmul", &fast_eigen_mult, py::arg("m1").noconvert(), py::arg("m2").noconvert(), py::arg("num_slices") = 1000);
  }
} // namespace thirdai::matrix::python

// TODO(all): Figure out naming convention for python exposed classes and
// methods
// TODO(any): Add docstrings to methods
// TODO(any): Can we remove redudancy in the bindings?
PYBIND11_MODULE(thirdai, m) {  // NOLINT

#ifndef __clang__
  m.def("set_global_num_threads", &omp_set_num_threads,
        py::arg("max_num_threads"));
#endif

  // Per pybind11 docs breaking up the construction of bindings in this way
  // could speed up build times. See below for more info:
  // https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-reduce-the-build-time
  thirdai::dataset::python::createDatasetSubmodule(m);

  thirdai::hashing::python::createHashingSubmodule(m);

  thirdai::bolt::python::createBoltSubmodule(m);

  thirdai::search::python::createSearchSubmodule(m);

  thirdai::matrix::python::createMatrixSubmodule(m);
}