#include "HashingPython.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <hashing/src/DWTA.h>
#include <hashing/src/DensifiedMinHash.h>
#include <hashing/src/FastSRP.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

namespace thirdai::hashing::python {

void createHashingSubmodule(py::module_& module) {
  auto hashing_submodule = module.def_submodule("hashing");

  py::class_<HashFunction, std::shared_ptr<HashFunction>>(
      hashing_submodule, "HashFunction",
      "Represents an abstract hash function that maps input DenseVectors and "
      "SparseVectors to sets of positive integers")
      .def("get_num_tables", &HashFunction::numTables,
           "Returns the number of hash tables in this hash function, which is "
           "equivalently the number of hashes that get returned by the "
           "function for each input.")
      .def("get_range", &HashFunction::range,
           "All hashes returned from this function will be >= 0 and <= "
           "get_range().");

  py::class_<DensifiedMinHash, std::shared_ptr<DensifiedMinHash>, HashFunction>(
      hashing_submodule, "MinHash",
      "A concrete implementation of a HashFunction that performs an extremly "
      "efficient minhash. A statistical estimator of jaccard similarity.")
      .def(py::init<uint32_t, uint32_t, uint32_t>(),
           py::arg("hashes_per_table"), py::arg("num_tables"),
           py::arg("range"));

  py::class_<FastSRP, std::shared_ptr<FastSRP>, HashFunction>(
      hashing_submodule, "SignedRandomProjection",
      "A concrete implementation of a HashFunction that performs an extremly "
      "efficient signed random projection. A statistical estimator of cossine "
      "similarity.")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
           py::arg("input_dim"), py::arg("hashes_per_table"),
           py::arg("num_tables"), py::arg("range") = UINT32_MAX);

  py::class_<DWTAHashFunction, std::shared_ptr<DWTAHashFunction>, HashFunction>(
      hashing_submodule, "DWTA")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                    std::optional<uint32_t>>(),
           py::arg("input_dim"), py::arg("hashes_per_table"),
           py::arg("num_tables"), py::arg("range_pow"), py::arg("binsize"),
           py::arg("permutations"))
      .def("save", &DWTAHashFunction::save, py::arg("filename"))
      .def_static("load", &DWTAHashFunction::load, py::arg("filename"))
      .def(bolt::python::getPickleFunction<DWTAHashFunction>());
}
}  // namespace thirdai::hashing::python