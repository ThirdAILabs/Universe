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
#include <bolt/src/networks/DLRM.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
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
        train_data.dataset, train_labels.dataset, loss_fn, learning_rate,
        epochs, rehash, rebuild, metric_names, verbose);

#if defined(__linux__) || defined(__APPLE__)

    // Restoring the old signal handler here
    std::signal(SIGINT, old_signal_handler);

#endif

    return metrics;
  }

  py::tuple predict(
      const py::object& data, const py::object& labels, uint32_t batch_size = 0,
      bool use_sparse_inference = false,
      const std::vector<std::string>& metrics = {}, bool verbose = true,
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max()) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));

    auto test_data = convertPyObjectToBoltDataset(data, batch_size, false);

    BoltDatasetNumpyContext test_labels;
    if (!labels.is_none()) {
      test_labels = convertPyObjectToBoltDataset(labels, batch_size, true);
    }

    uint32_t num_samples = test_data.dataset->len();

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
        test_data.dataset, test_labels.dataset, active_neurons, activations,
        use_sparse_inference, metrics, verbose, batch_limit);

    py::dict py_metric_data = py::cast(metric_data);

    return constructPythonInferenceTuple(std::move(py_metric_data), num_samples,
                                         inference_output_dim, active_neurons,
                                         activations);
  }

  void saveForInference(const std::string& filename) {
    this->save(filename, /* shallow= */ true);
  }

  /**
   * To save without optimizer, shallow=true
   */
  void save(const std::string& filename, bool shallow) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    this->setShallowSave(shallow);
    oarchive(*this);
  }

  void checkpoint(const std::string& filename) {
    if (this->anyLayerShallow()) {
      throw std::logic_error("Trying to checkpoint a model with no optimizer");
    }
    this->save(filename, /* shallow= */ false);
  }

  /**
   * Removes the optimizer state for the network by setting layers to shallow
   */
  void trimForInference() { this->setShallow(true); }

  /**
   * If any of the layer is shallow, that is without an optimzier, reinitiliaze
   * optimizer for that layer to 0.
   */
  void reinitOptimizerForTraining() { this->setShallow(false); }

  /**
   * If any layer in the model is shallow i.e, has uninitialized optimizer,
   * return false
   */
  bool isReadyForTraining() { return !this->anyLayerShallow(); }

  static std::unique_ptr<PyNetwork> load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
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

  void setTrainable(uint32_t layer_index, bool trainable) {
    _layers.at(layer_index)->setTrainable(trainable);
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
      const dataset::BoltDatasetPtr& test_labels, bool use_sparse_inference,
      const std::vector<std::string>& metrics = {}, bool verbose = true,
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max()) {
    uint32_t num_samples = test_data->len();
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

    auto metric_data =
        DLRM::predict(test_data, test_labels, active_neurons, activations,
                      use_sparse_inference, metrics, verbose, batch_limit);
    py::dict py_metric_data = py::cast(metric_data);

    return constructPythonInferenceTuple(std::move(py_metric_data), num_samples,
                                         inference_output_dim, active_neurons,
                                         activations);
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
    BoltVector vec = dataset::python::parseSentenceToBoltVector(
        sentence, /* seed= */ 341, _model->getInputDim());
    _model->forward(0, vec, _output, /* labels= */ nullptr);
    return _output.activations[1];
  }

 private:
  std::unique_ptr<PyNetwork> _model;
  BoltVector _output;
};

}  // namespace thirdai::bolt::python

CEREAL_REGISTER_TYPE(thirdai::bolt::python::PyNetwork)