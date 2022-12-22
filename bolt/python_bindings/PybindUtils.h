#pragma once

#include <bolt/src/metrics/MetricAggregator.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <dataset/src/Datasets.h>
#include <pybind11/cast.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt::python {

void printMemoryWarning(uint64_t num_samples, uint64_t inference_dim);

void printCopyWarning(const std::string& array_name, const py::str& dtype_recv,
                      const std::string& dtype_expected);

void biasDimensionCheck(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& biases,
    uint64_t dim, const std::string& matrix_type);

void weightDimensionCheck(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        new_weights,
    uint64_t dim, uint64_t prev_dim, const std::string& matrix_type = "");

void layerIndexCheck(uint32_t layer_index, uint32_t num_layers);

//  Checks that the dimensions of the given numpy array match the expected
//  dimensions.
void checkNumpyArrayDimensions(
    const std::vector<uint32_t>& expected_dimensions,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        numpy_array);

void allocateActivations(uint64_t num_samples, uint64_t inference_dim,
                         uint32_t** active_neurons, float** activations,
                         bool output_sparse);

// Takes in the activations arrays (if they were allocated) and returns the
// python tuple containing the metrics computed, along with the activations
// and active neurons if those are not nullptrs. Note that just the
// active_neuron pointer can be null if the output is dense. The
// activation_handle object is a python object that owns the data for
// the activations array, and likewise for the active_neuron_handle.
py::tuple constructPythonInferenceTuple(py::dict&& py_metric_data,
                                        uint32_t num_samples,
                                        uint32_t inference_dim,
                                        const float* activations,
                                        const uint32_t* active_neurons,
                                        const py::object& activation_handle,
                                        const py::object& active_neuron_handle);

// Helper method where the handles for active_neurons and activations are
// automatically constructed assuming no other c++ object owns their memory
py::tuple constructPythonInferenceTuple(py::dict&& py_metric_data,
                                        uint32_t num_samples,
                                        uint32_t inference_dim,
                                        const float* activations,
                                        const uint32_t* active_neurons);

py::array_t<float, py::array::c_style | py::array::forcecast>
denseBoltVectorToNumpy(const BoltVector& output);

// This struct is used to wrap a char* into a stream, see
// https://stackoverflow.com/questions/7781898/get-an-istream-from-a-char
struct Membuf : std::streambuf {
  Membuf(char* begin, char* end) { this->setg(begin, begin, end); }
};

// This function defines the pickle method for a type, assuming that type
// has a static load method that takes in an istream and a save method that
// takes in an ostream.
// Pybind pickling reference:
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support
// py::bytes -> char*:
// https://github.com/pybind/pybind11/issues/2517
// char* -> istream:
// https://stackoverflow.com/questions/7781898/get-an-istream-from-a-char
template <typename SERIALIZE_T>
pybind11::detail::initimpl::pickle_factory<
    std::function<py::bytes(const SERIALIZE_T&)>,
    std::function<std::shared_ptr<SERIALIZE_T>(
        py::bytes)>> inline static getPickleFunction() {
  return py::pickle<std::function<py::bytes(const SERIALIZE_T&)>,
                    std::function<std::shared_ptr<SERIALIZE_T>(py::bytes)>>(
      [](const SERIALIZE_T& model) {
        std::stringstream ss;
        model.save_stream(ss);
        std::string binary_model = ss.str();
        return py::bytes(binary_model);
      },
      [](const py::bytes& binary_model_python) {  // __setstate__
        py::buffer_info info(py::buffer(binary_model_python).request());
        char* binary_model = reinterpret_cast<char*>(info.ptr);
        Membuf sbuf(binary_model, binary_model + info.size);
        std::istream in(&sbuf);
        return SERIALIZE_T::load_stream(in);
      });
}

// This redirects std::out and std::err to pythons output and error streams,
// respectively, so that prints followed by a flush are immediately visible,
// even in notebooks. See the following link for more details:
// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html#capturing-standard-output-from-ostream
using OutputRedirect =
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>;

}  // namespace thirdai::bolt::python
