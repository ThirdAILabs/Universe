#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "ConversionUtils.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/networks/DistributedModel.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <_types/_uint32_t.h>
#include <compression/src/CompressedVector.h>
#include <compression/src/DragonVector.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <dataset/src/DatasetLoaders.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <csignal>
#include <exception>
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

void createBoltSubmodule(py::module_& module);

// Returns true on success and false on allocation failure.
void allocateActivations(uint64_t num_samples, uint64_t inference_dim,
                         uint32_t** active_neurons, float** activations,
                         bool output_sparse);

class PyNetwork final : public FullyConnectedNetwork {
 public:
  PyNetwork(SequentialConfigList configs, uint64_t input_dim)
      : FullyConnectedNetwork(std::move(configs), input_dim) {}

  MetricData train(dataset::BoltDatasetPtr& data,
                   const dataset::BoltDatasetPtr& labels,
                   const LossFunction& loss_fn, float learning_rate,
                   uint32_t epochs, uint32_t rehash = 0, uint32_t rebuild = 0,
                   const std::vector<std::string>& metric_names = {},
                   bool verbose = false) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));

/**Overriding the SIG_INT exception handler to exit the program once
 * CTRL+C is pressed by the user to interrupt the execution of any long
 * running code.
 */
#if defined(__linux__) || defined(__APPLE__)

    auto handler = [](int code) {
      std::cout << "Caught a keyboard interrupt with code: " << code
                << std::endl;
      std::cout << "Gracefully shutting down the program!" << std::endl;
      exit(code);
    };

    /**
     * signal() function returns the current signal handler.
     */
    using sighandler_t = void (*)(int); /* for convenience */
    sighandler_t old_signal_handler = std::signal(SIGINT, handler);

#endif

    /**
     * For windows signal() function from csignal doesnot works for raising
     * CTRL+C interrupts Look into below link for further information.
     * https://stackoverflow.com/questions/54362699/windows-console-signal-handling-for-subprocess-c
     */

    MetricData metrics = FullyConnectedNetwork::train(
        data, labels, loss_fn, learning_rate, epochs, rehash, rebuild,
        metric_names, verbose);

#if defined(__linux__) || defined(__APPLE__)

    // Restoring the old signal handler here
    std::signal(SIGINT, old_signal_handler);

#endif

    return metrics;
  }

  py::tuple getInputGradients(
      dataset::BoltDatasetPtr& data, const LossFunction& loss_fn,
      bool best_index = true,
      const std::vector<uint32_t>& required_labels = std::vector<uint32_t>()) {
    auto gradients = FullyConnectedNetwork::getInputGradients(
        data, loss_fn, best_index, required_labels);

    if (gradients.second == std::nullopt) {
      return py::cast(gradients.first);
    }
    return py::cast(gradients);
  }

  py::tuple predict(
      const dataset::BoltDatasetPtr& data,
      const dataset::BoltDatasetPtr& labels, bool use_sparse_inference = false,
      const std::vector<std::string>& metrics = {}, bool verbose = true,
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max()) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));

    uint32_t num_samples = data->len();

    uint64_t inference_output_dim = getInferenceOutputDim(use_sparse_inference);
    bool output_sparse = inference_output_dim < getOutputDim();

    // Declare pointers to memory for activations and active neurons, if the
    // allocation succeeds this will be assigned valid addresses by the
    // allocateActivations function. Otherwise the nullptr will indicate that
    // activations are not being computed.
    uint32_t* active_neurons = nullptr;
    float* activations = nullptr;

    allocateActivations(num_samples, inference_output_dim, &active_neurons,
                        &activations, output_sparse);

    auto metric_data = FullyConnectedNetwork::predict(
        data, labels, active_neurons, activations, use_sparse_inference,
        metrics, verbose, batch_limit);

    py::dict py_metric_data = py::cast(metric_data);

    return constructPythonInferenceTuple(std::move(py_metric_data), num_samples,
                                         inference_output_dim, activations,
                                         active_neurons);
  }

  /**
   * To save without optimizer, shallow=true
   */
  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<PyNetwork> load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<PyNetwork> deserialize_into(new PyNetwork());
    iarchive(*deserialize_into);
    return deserialize_into;
  }

  py::array_t<float> getWeights(uint32_t layer_index) {
    layerIndexCheck(layer_index, _num_layers);
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

  void setTrainable(uint32_t layer_index, bool trainable) {
    _layers.at(layer_index)->setTrainable(trainable);
  }

  void setWeights(
      uint32_t layer_index,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          new_weights) {
    uint64_t dim = _layers.at(layer_index)->getDim();
    uint64_t prev_dim =
        (layer_index > 0) ? _layers.at(layer_index - 1)->getDim() : _input_dim;

    weightDimensionCheck(new_weights, dim, prev_dim,
                         /* matrix type */ "weight matrix");

    _layers.at(layer_index)->setWeights(new_weights.data());
  }

  void setBiases(
      uint32_t layer_index,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          new_biases) {
    uint64_t dim = _layers.at(layer_index)->getDim();

    biasDimensionCheck(new_biases, dim, /* matrix type */ "bias matrix");
    _layers.at(layer_index)->setBiases(new_biases.data());
  }

  py::array_t<float> getBiases(uint32_t layer_index) {
    layerIndexCheck(layer_index, _num_layers);

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

  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  PyNetwork() : FullyConnectedNetwork(){};
};

class DistributedPyNetwork final : public DistributedModel {
 public:
  DistributedPyNetwork(SequentialConfigList configs, uint64_t input_dim)
      : DistributedModel(std::move(configs), input_dim) {}

  uint32_t prepareNodeForDistributedTraining(
      dataset::BoltDatasetPtr& data, const dataset::BoltDatasetPtr& labels,
      uint32_t rehash = 0, uint32_t rebuild = 0, bool verbose = false) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));

    uint32_t num_of_batches =
        DistributedModel::prepareNodeForDistributedTraining(
            data, labels, rehash, rebuild, verbose);

    return num_of_batches;
  }

  py::tuple predictSingleNode(
      const dataset::BoltDatasetPtr& data,
      const dataset::BoltDatasetPtr& labels, bool use_sparse_inference = false,
      const std::vector<std::string>& metrics = {}, bool verbose = true,
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max()) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));

    uint32_t num_samples = data->len();

    uint64_t inference_output_dim =
        DistributedModel::getInferenceOutputDim(use_sparse_inference);
    bool output_sparse =
        inference_output_dim < DistributedModel::getOutputDim();

    // Declare pointers to memory for activations and active neurons, if the
    // allocation succeeds this will be assigned valid addresses by the
    // allocateActivations function. Otherwise the nullptr will indicate that
    // activations are not being computed.
    uint32_t* active_neurons = nullptr;
    float* activations = nullptr;

    allocateActivations(num_samples, inference_output_dim, &active_neurons,
                        &activations, output_sparse);

    auto metric_data = DistributedModel::predict(
        data, labels, active_neurons, activations, use_sparse_inference,
        metrics, verbose, batch_limit);

    py::dict py_metric_data = py::cast(metric_data);

    return constructPythonInferenceTuple(std::move(py_metric_data), num_samples,
                                         inference_output_dim, activations,
                                         active_neurons);
  }

  py::array_t<float> getWeights(uint32_t layer_index) {
    layerIndexCheck(layer_index, DistributedModel::numLayers());

    float* mem = DistributedModel::getWeights(layer_index);

    py::capsule free_when_done(
        mem, [](void* ptr) { delete static_cast<float*>(ptr); });

    size_t dim = DistributedModel::getDim(layer_index);
    size_t prev_dim = (layer_index > 0)
                          ? DistributedModel::getDim(layer_index - 1)
                          : DistributedModel::getInputDim();

    return py::array_t<float>({dim, prev_dim},
                              {prev_dim * sizeof(float), sizeof(float)}, mem,
                              free_when_done);
  }

  void setWeights(
      uint32_t layer_index,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          new_weights) {
    uint64_t dim = DistributedModel::getDim(layer_index);
    uint64_t prev_dim = (layer_index > 0)
                            ? DistributedModel::getDim(layer_index - 1)
                            : DistributedModel::getInputDim();

    weightDimensionCheck(new_weights, dim, prev_dim,
                         /* matrix type*/ "weight matrix");

    DistributedModel::setWeights(layer_index, new_weights.data());
  }

  void setWeightGradients(
      uint32_t layer_index,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          new_weights_gradients) {
    uint64_t dim = DistributedModel::getDim(layer_index);
    uint64_t prev_dim = (layer_index > 0)
                            ? DistributedModel::getDim(layer_index - 1)
                            : DistributedModel::getInputDim();

    weightDimensionCheck(new_weights_gradients, dim, prev_dim,
                         /* matrix_type */ "weight gradient matrix");
    DistributedModel::setWeightGradients(layer_index,
                                         new_weights_gradients.data());
  }

  void setBiases(
      uint32_t layer_index,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          new_biases) {
    uint64_t dim = DistributedModel::getDim(layer_index);
    biasDimensionCheck(new_biases, dim, /* matrix type */ "bias matrix");

    DistributedModel::setBiases(layer_index, new_biases.data());
  }

  void setBiasesGradients(
      uint32_t layer_index,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          new_biases_gradients) {
    uint64_t dim = DistributedModel::getDim(layer_index);

    biasDimensionCheck(new_biases_gradients, dim,
                       /* matrix_type */ "bias gradient matrix");
    DistributedModel::setBiasesGradients(layer_index,
                                         new_biases_gradients.data());
  }

  py::array_t<float> getBiases(uint32_t layer_index) {
    layerIndexCheck(layer_index, DistributedModel::numLayers());

    float* mem = DistributedModel::getBiases(layer_index);

    py::capsule free_when_done(
        mem, [](void* ptr) { delete static_cast<float*>(ptr); });

    size_t dim = DistributedModel::getDim(layer_index);

    return py::array_t<float>({dim}, {sizeof(float)}, mem, free_when_done);
  }

  /*
   * We dont need to free the float* mem we are getting from the getLayerData
   * in the next two function as the pointer directly points to raw data of the
   * vector(due to vector.data()) and would be freed by the destructor of vector
   * itself.
   */

  // TODO(pratik): Rather than just passing vector.data(),
  // we should create the array with a py::object generated from a
  // py::cast(this)

  py::array_t<float> getBiasesGradients(uint32_t layer_index) {
    layerIndexCheck(layer_index, DistributedModel::numLayers());

    float* mem = DistributedModel::getBiasesGradient(layer_index);

    size_t dim = DistributedModel::getDim(layer_index);

    return py::array_t<float>({dim}, {sizeof(float)}, mem);
  }

  py::array_t<float> getWeightsGradients(uint32_t layer_index) {
    layerIndexCheck(layer_index, DistributedModel::numLayers());

    float* mem = DistributedModel::getWeightsGradient(layer_index);

    size_t dim = DistributedModel::getDim(layer_index);
    size_t prev_dim = (layer_index > 0)
                          ? DistributedModel::getDim(layer_index - 1)
                          : DistributedModel::getInputDim();

    return py::array_t<float>({dim, prev_dim},
                              {prev_dim * sizeof(float), sizeof(float)}, mem);
  }

  py::tuple getCompressedGradients(const std::string& compression_scheme,
                                   uint32_t layer_index,
                                   float compression_density,
                                   bool sketch_biases, int seed_for_hashing) {
    size_t dim = DistributedModel::getDim(layer_index);
    size_t prev_dim = (layer_index > 0)
                          ? DistributedModel::getDim(layer_index - 1)
                          : DistributedModel::getInputDim();

    if (sketch_biases) {
      if (compression_scheme == "dragon") {
        compression::DragonVector<float> dragon_sketch =
            compression::DragonVector<float>(
                DistributedModel::getBiasesGradient(layer_index),
                compression_density, uint32_t(dim), seed_for_hashing);

        return py::make_tuple(
            py::array_t<uint32_t>(py::cast(dragon_sketch.getIndices())),
            py::array_t<float>(py::cast(dragon_sketch.getValues())));
      }
    } else {
      if (compression_scheme == "dragon") {
        compression::DragonVector<float> dragon_sketch =
            compression::DragonVector<float>(
                DistributedModel::getWeightsGradient(layer_index),
                compression_density, uint32_t(dim * prev_dim),
                seed_for_hashing);

        std::cout << "printing dragon indices: " << std::endl;
        for (auto x : dragon_sketch.getIndices()) {
          std::cout << x << " ";
        }
        std::cout << std::endl;

        std::cout << "printing dragon gradients: " << std::endl;
        for (auto x : dragon_sketch.getValues()) {
          std::cout << x << " ";
        }
        std::cout << std::endl;

        return py::make_tuple(
            py::array_t<uint32_t>(py::cast(dragon_sketch.getIndices())),
            py::array_t<float>(py::cast(dragon_sketch.getValues())));
      }
    }
    throw std::logic_error("Not a valid compression scheme specified");
    return py::make_tuple(0, 0);
  }
};

class SentimentClassifier {
 public:
  explicit SentimentClassifier(const std::string& model_path) {
    _model = PyNetwork::load(model_path);
    _model->initializeNetworkState(/* batch_size= */ 1,
                                   /* use_sparsity= */ true);
    if (_model->getOutputDim() != 2) {
      throw std::invalid_argument(
          "Expected model output dim to be 2 for sentiment classifier.");
    }
    _output = BoltVector(/* l= */ 2, /* is_dense= */ true);
  }

  float predictSentiment(const std::string& sentence) {
    BoltVector vec = dataset::TextEncodingUtils::computeUnigrams(
        sentence, _model->getInputDim());
    _model->forward(0, vec, _output, /* labels= */ nullptr);
    return _output.activations[1];
  }

 private:
  std::unique_ptr<PyNetwork> _model;
  BoltVector _output;
};

}  // namespace thirdai::bolt::python

CEREAL_REGISTER_TYPE(thirdai::bolt::python::PyNetwork)