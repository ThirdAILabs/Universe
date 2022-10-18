#include "DatasetPython.h"
#include <new_dataset/src/Dataset.h>
#include <new_dataset/src/NumpyDataset.h>
#include <pybind11/operators.h>

namespace thirdai::dataset::python {

namespace py = pybind11;

void createNewDatasetSubmodule(py::module_& module) {
  // Everything in this submodule is exposed to users.
  auto dataset_submodule = module.def_submodule("new_dataset");

  // TODO(Josh): Add other numpy methods
  dataset_submodule.def("from_np", &numpy::denseNumpyToDataset,
                        py::arg("data"));

  // For reference on binding iterable C++ types in python, see
  // https://github.com/pybind/pybind11/blob/master/tests/test_sequences_and_iterators.cpp
  py::class_<Dataset, DatasetPtr>(dataset_submodule, "Dataset")
      .def("__getitem__",
           [](const Dataset& d, size_t i) {
             if (i >= d.len()) {
               throw py::index_error();
             }
             return d[i];
           })
      .def("__setitem__",
           [](Dataset& d, size_t i, BoltVector v) {
             if (i >= d.len()) {
               throw py::index_error();
             }
             d.set(i, std::move(v));
           })
      .def("__len__", &Dataset::len)
      /// Optional sequence protocol operations
      .def(
          "__iter__",
          [](const Dataset& d) {
            return py::make_iterator(d.begin(), d.end());
          },
          py::keep_alive<
              0, 1>() /* Essential: keep object alive while iterator exists */)
      .def("__getitem__",
           [](const Dataset& d, const py::slice& slice) -> DatasetPtr {
             size_t start = 0, stop = 0, step = 0, slicelength = 0;
             if (!slice.compute(d.len(), &start, &stop, &step, &slicelength)) {
               throw py::error_already_set();
             }
             if (step != 1) {
               throw std::invalid_argument(
                   "Dataset only supports slices with step size 1");
             }
             return d.slice(start, stop);
           })
      .def("__setitem__",
           [](Dataset& d, const py::slice& slice, const Dataset& value) {
             size_t start = 0, stop = 0, step = 0, slicelength = 0;
             if (!slice.compute(d.len(), &start, &stop, &step, &slicelength)) {
               throw py::error_already_set();
             }
             if (slicelength != value.len()) {
               throw std::runtime_error(
                   "Left and right hand size of slice assignment have "
                   "different sizes!");
             }
             for (size_t i = 0; i < slicelength; ++i) {
               d[start] = value[i];
               start += step;
             }
           });

  py::class_<numpy::NumpyDataset, Dataset, numpy::NumpyDatasetPtr>(
      dataset_submodule, "NumpyDataset");  // NOLINT

  // TODO(Josh): We can always add various other python specific methods later:
  //  contains, reverse, append, concatenate, etc.
}

}  // namespace thirdai::dataset::python
