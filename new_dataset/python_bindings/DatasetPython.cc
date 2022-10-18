#include <dataset/src/NumpyDataset.h>
#include <new_dataset/src/Dataset.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace thirdai::dataset::python {

namespace py = pybind11;

void createDatasetSubmodule(py::module_& module) {
  // Everything in this submodule is exposed to users.
  auto dataset_submodule = module.def_submodule("new_dataset");

  dataset_submodule.def("from_np", &numpy::numpyToBoltVectorDataset,
                        py::arg("data"));

  py::class_<Dataset>(dataset_submodule, "Dataset")
      .def(py::init<size_t>())
      .def(py::init<const std::vector<float>&>())
      /// Bare bones interface
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
             if (slicelength != 1) {
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

    // TODO(Josh): We can always add various other helper methods later:
    //  contains, reverse, append, concatenate, etc.
}

}  // namespace thirdai::dataset::python
