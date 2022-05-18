#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/networks/DLRM.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <pybind11/cast.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace thirdai::bolt::python {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
using NumpyArrayPtr = std::unique_ptr<NumpyArray<T>>;

void createBoltSubmodule(py::module_& module);

// Returns true on success and false on allocation failure.
bool allocateActivations(uint32_t num_samples, uint32_t inference_dim,
                         uint32_t** active_neurons, float** activations,
                         bool output_sparse);

// Takes in the activations arrays (if they were allocated) and returns the
// correct python tuple containing the activations (and active neurons if
// sparse) and the metrics computed.
py::tuple constructNumpyArrays(py::dict&& py_metric_data, uint32_t num_samples,
                               uint32_t inference_dim, uint32_t* active_neurons,
                               float* activations, bool output_sparse,
                               bool alloc_success);

static bool isBoltDataset(const py::object& obj) {
  return py::str(obj.get_type())
      .equal(py::str("<class 'thirdai._thirdai.dataset.BoltDataset'>"));
}

static bool isTuple(const py::object& obj) {
  return py::str(obj.get_type()).equal(py::str("<class 'tuple'>"));
}

static bool isNumpyArray(const py::object& obj) {
  return py::str(obj.get_type()).equal(py::str("<class 'numpy.ndarray'>"));
}

struct BoltDatasetWithNumpyContext {
  dataset::BoltDatasetPtr dataset;
  std::optional<NumpyArray<uint32_t>> indices;
  std::optional<NumpyArray<float>> values;
  std::optional<NumpyArray<uint32_t>> offsets;
};

class PyNetwork final : public FullyConnectedNetwork {
 public:
  PyNetwork(SequentialConfigList configs, uint64_t input_dim)
      : FullyConnectedNetwork(std::move(configs), input_dim) {}

  MetricData train(const py::object& data, const py::object& labels,
                   const LossFunction& loss_fn, float learning_rate,
                   uint32_t epochs, uint32_t batch_size = 0,
                   uint32_t rehash = 0, uint32_t rebuild = 0,
                   const std::vector<std::string>& metric_names = {},
                   bool verbose = false) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));
    auto train_data = convertPyObjectToBoltDataset(data, batch_size, false);

    auto train_labels = convertPyObjectToBoltDataset(labels, batch_size, true);

    // const uint32_t* x =
    //     data.cast<py::tuple>()[0].cast<NumpyArray<uint32_t>>().data();
    // std::cout << "ARRAY POINTER: " << x << std::endl;
    // std::cout << "ARRAY:" << std::endl;
    // for (uint32_t i = 0;
    //      i < train_data.dataset->len() * train_data.dataset->at(0)[0].len;
    //      i++) {
    //   std::cout << x[i] << " ";
    // }

    // std::cout << std::endl;
    // std::cout << "INDICES:" << std::endl;
    // std::cout << "INDICES POINTER: "
    //           << train_data.dataset->at(0)[0].active_neurons << std::endl;
    // for (uint32_t i = 0;
    //      i < train_data.dataset->len() * train_data.dataset->at(0)[0].len;
    //      i++) {
    //   std::cout << train_data.dataset->at(0)[0].active_neurons[i] << " ";
    // }
    // std::cout << std::endl;

    return FullyConnectedNetwork::train(
        train_data.dataset, train_labels.dataset, loss_fn, learning_rate,
        epochs, rehash, rebuild, metric_names, verbose);
  }

  py::tuple predict(
      const py::object& data, const py::object& labels, uint32_t batch_size = 0,
      const std::vector<std::string>& metrics = {}, bool verbose = true,
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max()) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));

    auto test_data = convertPyObjectToBoltDataset(data, batch_size, false);

    BoltDatasetWithNumpyContext test_labels = {nullptr, std::nullopt,
                                               std::nullopt, std::nullopt};
    if (!labels.is_none()) {
      test_labels = convertPyObjectToBoltDataset(labels, batch_size, true);
    }

    uint32_t num_samples = test_data.dataset->len();

    bool output_sparse = getInferenceOutputDim() < getOutputDim();

    // Declare pointers to memory for activations and active neurons, if the
    // allocation succeeds this will be assigned valid addresses by the
    // allocateActivations function. Otherwise the nullptr will indicate that
    // activations are not being computed.
    uint32_t* active_neurons = nullptr;
    float* activations = nullptr;

    bool alloc_success =
        allocateActivations(num_samples, getInferenceOutputDim(),
                            &active_neurons, &activations, output_sparse);

    auto metric_data = FullyConnectedNetwork::predict(
        test_data.dataset, test_labels.dataset, active_neurons, activations, metrics, verbose,
        batch_limit);

    py::dict py_metric_data = py::cast(metric_data);

    return constructNumpyArrays(std::move(py_metric_data), num_samples,
                                getInferenceOutputDim(), active_neurons,
                                activations, output_sparse, alloc_success);
  }

  void save(const std::string& filename) {
    std::ofstream filestream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<PyNetwork> load(const std::string& filename) {
    std::ifstream filestream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<PyNetwork> deserialize_into(new PyNetwork());
    iarchive(*deserialize_into);
    return deserialize_into;
  }

  py::array_t<float> getWeights(uint32_t layer_index) {
    if (layer_index >= _num_layers) {
      return py::none();
    }

    float* mem = _layers[layer_index]->getWeights();

    py::capsule free_when_done(
        mem, [](void* ptr) { delete static_cast<float*>(ptr); });

    size_t dim = _layers.at(layer_index)->getDim();
    size_t prev_dim =
        (layer_index > 0) ? _layers.at(layer_index - 1)->getDim() : _input_dim;

    return py::array_t<float>({dim, prev_dim},
                              {prev_dim * sizeof(float), sizeof(float)}, mem,
                              free_when_done);
  }

  void setWeights(
      uint32_t layer_index,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          new_weights) {
    int64_t dim = _layers.at(layer_index)->getDim();
    int64_t prev_dim =
        (layer_index > 0) ? _layers.at(layer_index - 1)->getDim() : _input_dim;

    if (new_weights.ndim() != 2) {
      std::stringstream err;
      err << "Expected weight matrix to have 2 dimensions, received matrix "
             "with "
          << new_weights.ndim() << " dimensions.";
      throw std::invalid_argument(err.str());
    }
    if (new_weights.shape(0) != dim || new_weights.shape(1) != prev_dim) {
      std::stringstream err;
      err << "Expected weight matrix to have dim (" << dim << ", " << prev_dim
          << ") received matrix with dim (" << new_weights.shape(0) << ", "
          << new_weights.shape(1) << ").";
      throw std::invalid_argument(err.str());
    }

    _layers.at(layer_index)->setWeights(new_weights.data());
  }

  void setBiases(
      uint32_t layer_index,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          new_biases) {
    int64_t dim = _layers.at(layer_index)->getDim();
    if (new_biases.ndim() != 1) {
      std::stringstream err;
      err << "Expected weight matrix to have 1 dimension, received matrix "
             "with "
          << new_biases.ndim() << " dimensions.";
      throw std::invalid_argument(err.str());
    }
    if (new_biases.shape(0) != dim) {
      std::stringstream err;
      err << "Expected weight matrix to have dim " << dim
          << " received matrix with dim " << new_biases.shape(0) << ".";
      throw std::invalid_argument(err.str());
    }

    _layers.at(layer_index)->setBiases(new_biases.data());
  }

  py::array_t<float> getBiases(uint32_t layer_index) {
    if (layer_index >= _num_layers) {
      return py::none();
    }

    float* mem = _layers[layer_index]->getBiases();

    py::capsule free_when_done(
        mem, [](void* ptr) { delete static_cast<float*>(ptr); });

    size_t dim = _layers.at(layer_index)->getDim();

    return py::array_t<float>({dim}, {sizeof(float)}, mem, free_when_done);
  }

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<FullyConnectedNetwork>(this));
  }

  static BoltDatasetWithNumpyContext convertTupleToBoltDataset(
      const py::object& obj, uint32_t batch_size) {
    if (batch_size == 0) {
      throw std::invalid_argument("No batch size provided.");
    }
    py::tuple tup = obj.cast<py::tuple>();
    if (tup.size() != 3) {
      throw std::invalid_argument(
          "Expected tuple of 3 numpy arrays (indices, values, offsets), "
          "received "
          "tuple of length: " +
          std::to_string(tup.size()));
    }

    if (!isNumpyArray(tup[0]) || !isNumpyArray(tup[1]) ||
        !isNumpyArray(tup[2])) {
      throw std::invalid_argument(
          "Expected tuple of 3 numpy arrays (indices, values, offsets), "
          "received non numpy array.");
    }

    NumpyArray<uint32_t> indices = tup[0].cast<NumpyArray<uint32_t>>();
    NumpyArray<float> values = tup[1].cast<NumpyArray<float>>();
    NumpyArray<uint32_t> offsets = tup[2].cast<NumpyArray<uint32_t>>();

    auto data = dataset::python::sparseBoltDatasetFromNumpy(
        indices, values, offsets, batch_size);

    return BoltDatasetWithNumpyContext{data, indices, values, offsets};
  }

  BoltDatasetWithNumpyContext convertNumpyArrayToBoltDataset(
      const py::object& obj, uint32_t batch_size, bool is_labels) {
    if (batch_size == 0) {
      throw std::invalid_argument("No batch size provided.");
    }

    NumpyArray<float> data = obj.cast<NumpyArray<float>>();
    if (is_labels && data.ndim() == 1) {
      auto array = data.cast<NumpyArray<uint32_t>>();
      auto dataset =
          dataset::python::categoricalLabelsFromNumpy(array, batch_size);
      return BoltDatasetWithNumpyContext{dataset, array, std::nullopt,
                                         std::nullopt};
    }

    uint32_t input_dim = data.ndim() == 1 ? 1 : data.shape(1);
    if (input_dim != getInputDim()) {
      throw std::invalid_argument("Cannot pass array with input dimension " +
                                  std::to_string(input_dim) +
                                  " to network with input dim " +
                                  std::to_string(getInputDim()));
    }

    auto dataset = dataset::python::denseBoltDatasetFromNumpy(data, batch_size);
    return {dataset, std::nullopt, data, std::nullopt};
  }

  BoltDatasetWithNumpyContext convertPyObjectToBoltDataset(
      const py::object& obj, uint32_t batch_size, bool is_labels) {
    if (isBoltDataset(obj)) {
      return {obj.cast<dataset::BoltDatasetPtr>(), std::nullopt, std::nullopt,
              std::nullopt};
    }
    if (isNumpyArray(obj)) {
      return convertNumpyArrayToBoltDataset(obj, batch_size, is_labels);
    }
    if (isTuple(obj)) {
      return convertTupleToBoltDataset(obj, batch_size);
    }

    throw std::invalid_argument(
        "Expected object of type BoltDataset, tuple, or numpy array (or None "
        "for "
        "test labels), received " +
        py::str(obj.get_type()).cast<std::string>());
  }

  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  PyNetwork() : FullyConnectedNetwork(){};
};

class PyDLRM final : public DLRM {
 public:
  PyDLRM(bolt::EmbeddingLayerConfig embedding_config,
         SequentialConfigList bottom_mlp_configs,
         SequentialConfigList top_mlp_configs, uint32_t input_dim)
      : DLRM(embedding_config, std::move(bottom_mlp_configs),
             std::move(top_mlp_configs), input_dim) {}

  py::tuple predict(
      const dataset::ClickThroughDatasetPtr& test_data,
      const dataset::BoltDatasetPtr& test_labels,
      const std::vector<std::string>& metrics = {}, bool verbose = true,
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max()) {
    uint32_t num_samples = test_data->len();

    bool output_sparse = getInferenceOutputDim() < getOutputDim();

    // Declare pointers to memory for activations and active neurons, if the
    // allocation succeeds this will be assigned valid addresses by the
    // allocateActivations function. Otherwise the nullptr will indicate that
    // activations are not being computed.
    uint32_t* active_neurons = nullptr;
    float* activations = nullptr;

    bool alloc_success =
        allocateActivations(num_samples, getInferenceOutputDim(),
                            &active_neurons, &activations, output_sparse);

    auto metric_data =
        DLRM::predict(test_data, test_labels, active_neurons, activations,
                      metrics, verbose, batch_limit);
    py::dict py_metric_data = py::cast(metric_data);

    return constructNumpyArrays(std::move(py_metric_data), num_samples,
                                getInferenceOutputDim(), active_neurons,
                                activations, output_sparse, alloc_success);
  }
};

}  // namespace thirdai::bolt::python

CEREAL_REGISTER_TYPE(thirdai::bolt::python::PyNetwork)
