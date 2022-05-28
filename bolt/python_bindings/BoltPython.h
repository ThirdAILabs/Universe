#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
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
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace thirdai::bolt::python {

void createBoltSubmodule(py::module_& module);

// Returns true on success and false on allocation failure.
bool allocateActivations(uint64_t num_samples, uint64_t inference_dim,
                         uint32_t** active_neurons, float** activations,
                         bool output_sparse);

// Takes in the activations arrays (if they were allocated) and returns the
// correct python tuple containing the activations (and active neurons if
// sparse) and the metrics computed.
py::tuple constructNumpyArrays(py::dict&& py_metric_data, uint32_t num_samples,
                               uint32_t inference_dim, uint32_t* active_neurons,
                               float* activations, bool output_sparse,
                               bool alloc_success);

class PyNetwork final : public FullyConnectedNetwork {
 public:
  PyNetwork(SequentialConfigList configs, uint64_t input_dim)
      : FullyConnectedNetwork(std::move(configs), input_dim) {}

  MetricData train(
      dataset::BoltDatasetPtr& train_data,
      const dataset::BoltDatasetPtr& train_labels,
      // Clang tidy is disabled for this line because it wants to pass by
      // reference, but shared_ptrs should not be passed by reference
      const LossFunction& loss_fn,  // NOLINT
      float learning_rate, uint32_t epochs, uint32_t rehash, uint32_t rebuild,
      const std::vector<std::string>& metric_names, bool verbose) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));
    return FullyConnectedNetwork::train(train_data, train_labels, loss_fn,
                                        learning_rate, epochs, rehash, rebuild,
                                        metric_names, verbose);
  }

  // Does not return py::array_t because this is consistent with the original
  // train method.
  MetricData trainWithDenseNumpyArray(
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          examples,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          labels,
      uint32_t batch_size, const LossFunction& loss_fn, float learning_rate,
      uint32_t epochs, uint32_t rehash, uint32_t rebuild,
      const std::vector<std::string>& metrics, bool verbose) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));
    auto batched_data =
        dataset::python::denseBoltDatasetFromNumpy(examples, batch_size);

    auto batched_labels =
        dataset::python::categoricalLabelsFromNumpy(labels, batch_size);

    // Prent clang tidy because it wants to pass the smart pointer by
    // reference
    return train(batched_data, batched_labels, loss_fn, learning_rate, epochs,
                 rehash,  // NOLINT
                 rebuild, metrics, verbose);
  }

  MetricData trainWithSparseNumpyArray(
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          x_idxs,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          x_vals,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          x_offsets,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          y_idxs,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          y_vals,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          y_offsets,
      uint32_t batch_size, const LossFunction& loss_fn, float learning_rate,
      uint32_t epochs, uint32_t rehash, uint32_t rebuild,
      const std::vector<std::string>& metrics, bool verbose) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));

    auto batched_data = thirdai::dataset::python::sparseBoltDatasetFromNumpy(
        x_idxs, x_vals, x_offsets, batch_size);

    auto batched_labels = thirdai::dataset::python::sparseBoltDatasetFromNumpy(
        y_idxs, y_vals, y_offsets, batch_size);
    // Prent clang tidy because it wants to pass the smart pointer by reference
    return train(batched_data, batched_labels, loss_fn, learning_rate, epochs,
                 rehash,  // NOLINT
                 rebuild, metrics, verbose);
  }

  py::tuple predict(
      const dataset::BoltDatasetPtr& test_data,
      const dataset::BoltDatasetPtr& test_labels,
      const std::vector<std::string>& metrics = {}, bool verbose = true,
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max()) {
    return predictImpl(test_data, test_labels, metrics, verbose, batch_limit);
  }

  py::tuple predictWithDenseNumpyArray(
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          examples,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          labels,
      uint32_t batch_size, const std::vector<std::string>& metrics,
      bool verbose, uint32_t batch_limit) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));

    auto batched_data = thirdai::dataset::python::denseBoltDatasetFromNumpy(
        examples, batch_size);

    auto batched_labels = thirdai::dataset::python::categoricalLabelsFromNumpy(
        labels, batch_size);

    return predictImpl(batched_data, batched_labels, metrics, verbose,
                       batch_limit);
  }

  py::tuple predictWithSparseNumpyArray(
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          x_idxs,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          x_vals,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          x_offsets,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          y_idxs,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          y_vals,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          y_offsets,
      uint32_t batch_size, const std::vector<std::string>& metrics,
      bool verbose, uint32_t batch_limit) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));

    auto batched_data = thirdai::dataset::python::sparseBoltDatasetFromNumpy(
        x_idxs, x_vals, x_offsets, batch_size);

    auto batched_labels = thirdai::dataset::python::sparseBoltDatasetFromNumpy(
        y_idxs, y_vals, y_offsets, batch_size);

    return predictImpl(batched_data, batched_labels, metrics, verbose,
                       batch_limit);
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

  py::tuple predictImpl(
      const dataset::BoltDatasetPtr& test_data,
      const dataset::BoltDatasetPtr& test_labels,
      const std::vector<std::string>& metrics = {}, bool verbose = true,
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max()) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));

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

    auto metric_data = FullyConnectedNetwork::predict(
        test_data, test_labels, active_neurons, activations, metrics, verbose,
        batch_limit);

    py::dict py_metric_data = py::cast(metric_data);

    return constructNumpyArrays(std::move(py_metric_data), num_samples,
                                getInferenceOutputDim(), active_neurons,
                                activations, output_sparse, alloc_success);
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
