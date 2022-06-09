#pragma once

#include <bolt/python_bindings/BoltPython.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <search/src/Blolt.h>
#include <cstdint>

namespace py = pybind11;

namespace thirdai::search::python {

class PyBlolt final : public Blolt {
 public:
  PyBlolt(uint64_t estimated_dataset_size, uint8_t num_classifiers,
          uint64_t input_dim)
      : Blolt(estimated_dataset_size, num_classifiers, input_dim) {}

  void index(const py::object& train_data_python,
             const py::object& entire_data_python, uint64_t batch_size) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));

    auto train_data_bolt = bolt::python::convertPyObjectToBoltDataset(
        train_data_python, batch_size, /* is_labels = */ false,
        /* network_input_dim = */ getInputDim());
    auto rest_of_data_bolt = bolt::python::convertPyObjectToBoltDataset(
        entire_data_python, batch_size, /* is_labels = */ false,
        /* network_input_dim = */ getInputDim());
    Blolt::index(train_data_bolt.dataset, rest_of_data_bolt.dataset);
  }

  std::vector<std::vector<uint64_t>> query(const py::object& query_batch_python,
                                           uint32_t top_k) {
    auto dataset_of_single_batch = bolt::python::convertPyObjectToBoltDataset(
        query_batch_python, /* batch_size = */ UINT32_MAX,
        /* is_labels = */ false,
        /* network_input_dim = */ getInputDim());
    return Blolt::query(dataset_of_single_batch.dataset->at(0), top_k);
  }
};

}  // namespace thirdai::search::python