#include "DatasetPython.h"
#include <new_dataset/src/Dataset.h>
#include <new_dataset/src/NumpyDataset.h>
#include <pybind11/operators.h>

namespace thirdai::dataset::python {

namespace py = pybind11;

void createNewDatasetSubmodule(py::module_& dataset_submodule) {
  // TODO(Josh): Add other numpy methods
  dataset_submodule.def("from_np", &numpy::denseNumpyToDataset,
                        py::arg("data"));

  // For reference on binding iterable C++ types in python, see
  // https://github.com/pybind/pybind11/blob/master/tests/test_sequences_and_iterators.cpp
  py::class_<Dataset, DatasetPtr>(dataset_submodule, "Dataset")
      .def("__getitem__",
           [](const Dataset& d, size_t i) {
             if (i >= d.size()) {
               throw py::index_error();
             }
             return d[i];
           })
      .def("__setitem__",
           [](Dataset& d, size_t i, BoltVector v) {
             if (i >= d.size()) {
               throw py::index_error();
             }
             d[i] = std::move(v);
           })
      .def("__len__", &Dataset::size)
      .def(
          "__iter__",
          [](const Dataset& d) {
            return py::make_iterator(d.begin(), d.end());
          },
          /*
           * This keep alive call ensures that the object with index 1 in the
           * function (the this pointer) is kept alive at least as long as the
           * object with index 0 (the returned iterator) is alive. See
           * https://pybind11.readthedocs.io/en/stable/advanced/functions.html#keep-alive
           */
          py::keep_alive<
              0, 1>() /* Essential: keep object alive while iterator exists */)
      .def("__getitem__",
           [](const Dataset& d, const py::slice& slice) -> DatasetPtr {
             size_t start = 0, stop = 0, step = 0, slicelength = 0;
             if (!slice.compute(d.size(), &start, &stop, &step, &slicelength)) {
               throw py::error_already_set();
             }
             if (step != 1) {
               throw std::invalid_argument(
                   "Dataset slices must have step size 1");
             }
             return d.slice(start, stop);
           })
      .def("__setitem__",
           [](Dataset& d, const py::slice& slice, const Dataset& value) {
             size_t start = 0, stop = 0, step = 0, slicelength = 0;
             if (!slice.compute(d.size(), &start, &stop, &step, &slicelength)) {
               /*
                * Computing the true slice indices is done in python, which
                * might throw an error. Pybind automatically saves that python
                * error and returns false from slice.compute in that case, so
                * here we can just call py::error_already_set to rethrow that
                * original python slice error.
                */
               throw py::error_already_set();
             }
             if (slicelength != value.size()) {
               throw std::runtime_error(
                   "Left and right hand size of slice assignment have "
                   "different sizes!");
             }
             for (size_t i = 0; i < slicelength; ++i) {
               d[start] = value[i];
               start += step;
             }
           })
      .def("copy", &Dataset::copy);

  py::class_<numpy::NumpyDataset, Dataset, numpy::NumpyDatasetPtr>(
      dataset_submodule, "NumpyDataset");  // NOLINT

  // TODO(Josh): We can always add various other python specific methods later:
  //  contains, reverse, append, concatenate, etc.
}

}  // namespace thirdai::dataset::python
