#include "HashingPython.h"

namespace thirdai::hashing::python {

void createHashingSubmodule(py::module_& module) {
  auto hashing_submodule = module.def_submodule("hashing");

  // TODO(josh): Add proper sparse data type

  py::class_<BloomFilter>(hashing_submodule, "BloomFilter")
      .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("num_hashes"),
           py::arg("requested_total_num_bits"), py::arg("input_dim"))
      .def("add", &BloomFilter::add,
           "Adds a vector of uint64s to the bloom filter. Must be of length "
           "input_dim.")
      .def("is_present", &BloomFilter::is_present,
           "Checks whether a vector was added to the filter. Has a non zero "
           "false positive rate and a zero false negative rate.");

  py::class_<HashFunction>(
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
  // TODO(josh): Add bindings for hashing numpy array and sparse data

  py::class_<DensifiedMinHash, HashFunction>(
      hashing_submodule, "MinHash",
      "A concrete implementation of a HashFunction that performs an extremly "
      "efficient minhash. A statistical estimator of jaccard similarity.")
      .def(py::init<uint32_t, uint32_t, uint32_t>(),
           py::arg("hashes_per_table"), py::arg("num_tables"),
           py::arg("range"));

  py::class_<FastSRP, HashFunction>(
      hashing_submodule, "SignedRandomProjection",
      "A concrete implementation of a HashFunction that performs an extremly "
      "efficient signed random projection. A statistical estimator of cossine "
      "similarity.")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
           py::arg("input_dim"), py::arg("hashes_per_table"),
           py::arg("num_tables"), py::arg("range") = UINT32_MAX);
}
}  // namespace thirdai::hashing::python