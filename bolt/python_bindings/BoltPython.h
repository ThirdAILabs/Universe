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

using thirdai::dataset::python::NumpyArray;

class BoltDatasetNumpyContext {
  /*
   * The purpose of this class is to make sure that a BoltDataset constructed
   * from a numpy array is memory safe by ensuring that the numpy arrays it is
   * constructed from cannot go out of scope while the dataset is in scope. This
   * problem arrises because if the numpy arrays passed in are not uint32 or
   * float32 then when we cast to that array type a copy will occur. This
   * resulting copy of the array will be a local copy, and thus when the method
   * constructing the dataset returns, the copy will go out of scope and the
   * dataset will be invalidated. This solves that issue.
   */
 public:
  dataset::BoltDatasetPtr dataset;

  explicit BoltDatasetNumpyContext()
      : dataset(nullptr),
        dataset_context_1(std::nullopt),
        dataset_context_2(std::nullopt) {}

  explicit BoltDatasetNumpyContext(dataset::BoltDatasetPtr&& _dataset)
      : dataset(_dataset),
        dataset_context_1(std::nullopt),
        dataset_context_2(std::nullopt) {}

  explicit BoltDatasetNumpyContext(NumpyArray<float>& examples,
                                   uint32_t batch_size)
      : dataset_context_2(std::nullopt) {
    dataset = dataset::python::denseBoltDatasetFromNumpy(examples, batch_size);
    dataset_context_1 = examples.request();
  }

  explicit BoltDatasetNumpyContext(NumpyArray<uint32_t>& labels,
                                   uint32_t batch_size)
      : dataset_context_2(std::nullopt) {
    dataset = dataset::python::categoricalLabelsFromNumpy(labels, batch_size);
    dataset_context_1 = labels.request();
  }

  explicit BoltDatasetNumpyContext(NumpyArray<uint32_t>& indices,
                                   NumpyArray<float>& values,
                                   NumpyArray<uint32_t>& offsets,
                                   uint32_t batch_size) {
    dataset = dataset::python::sparseBoltDatasetFromNumpy(indices, values,
                                                          offsets, batch_size);
    dataset_context_1 = indices.request();
    dataset_context_2 = values.request();
  }

 private:
  std::optional<py::buffer_info> dataset_context_1;
  std::optional<py::buffer_info> dataset_context_2;
};

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

static inline bool isBoltDataset(const py::object& obj) {
  return py::str(obj.get_type())
      .equal(py::str("<class 'thirdai._thirdai.dataset.BoltDataset'>"));
}

static inline bool isTuple(const py::object& obj) {
  return py::str(obj.get_type()).equal(py::str("<class 'tuple'>"));
}

static inline bool isNumpyArray(const py::object& obj) {
  return py::str(obj.get_type()).equal(py::str("<class 'numpy.ndarray'>"));
}

static inline py::str getDtype(const py::object& obj) {
  return py::str(obj.attr("dtype"));
}

static inline bool checkNumpyDtype(const py::object& obj,
                                   const std::string& type) {
  return getDtype(obj).equal(py::str(type));
}

static inline bool checkNumpyDtypeUint32(const py::object& obj) {
  return checkNumpyDtype(obj, "uint32");
}

static inline bool checkNumpyDtypeFloat32(const py::object& obj) {
  return checkNumpyDtype(obj, "float32");
}

static inline bool checkNumpyDtypeAnyInt(const py::object& obj) {
  return checkNumpyDtype(obj, "int32") || checkNumpyDtype(obj, "uint32") ||
         checkNumpyDtype(obj, "int64") || checkNumpyDtype(obj, "uint64");
}

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

  py::array_t<float> getInputGradients(
      const py::object& data, const LossFunction& loss_fn,
      const std::vector<uint32_t>& required_labels = std::vector<uint32_t>(),
      uint32_t batch_size = 256) {
    auto analysis_data = convertPyObjectToBoltDataset(data, batch_size, false);
    auto gradients = FullyConnectedNetwork::getInputGradients(
        analysis_data.dataset, loss_fn, required_labels);

    return py::cast(gradients);
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

    bool alloc_success =
        allocateActivations(num_samples, inference_output_dim, &active_neurons,
                            &activations, output_sparse);

    auto metric_data = FullyConnectedNetwork::predict(
        test_data.dataset, test_labels.dataset, active_neurons, activations,
        use_sparse_inference, metrics, verbose, batch_limit);

    py::dict py_metric_data = py::cast(metric_data);

    return constructNumpyArrays(std::move(py_metric_data), num_samples,
                                inference_output_dim, active_neurons,
                                activations, output_sparse, alloc_success);
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

  static void printCopyWarning(const std::string& array_name,
                               const py::str& dtype_recv,
                               const std::string& dtype_expected) {
    std::cout << "Warning: " << array_name << " array has dtype=" << dtype_recv
              << " but " << dtype_expected
              << " was expected. This will result in a copy of "
                 "the array in order to ensure type safety. Try specifying "
                 "the dtype of the array or use .astype(...)."
              << std::endl;
  }

  static BoltDatasetNumpyContext convertTupleToBoltDataset(
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

    if (!checkNumpyDtypeUint32(tup[0])) {
      printCopyWarning("indices", getDtype(tup[0]), "uint32");
    }
    if (!checkNumpyDtypeFloat32(tup[1])) {
      printCopyWarning("values", getDtype(tup[1]), "float32");
    }
    if (!checkNumpyDtypeUint32(tup[2])) {
      printCopyWarning("offsets", getDtype(tup[2]), "uint32");
    }

    NumpyArray<uint32_t> indices = tup[0].cast<NumpyArray<uint32_t>>();
    NumpyArray<float> values = tup[1].cast<NumpyArray<float>>();
    NumpyArray<uint32_t> offsets = tup[2].cast<NumpyArray<uint32_t>>();

    return BoltDatasetNumpyContext(indices, values, offsets, batch_size);
  }

  BoltDatasetNumpyContext convertNumpyArrayToBoltDataset(const py::object& obj,
                                                         uint32_t batch_size,
                                                         bool is_labels) {
    if (batch_size == 0) {
      throw std::invalid_argument("No batch size provided.");
    }

    if (is_labels && checkNumpyDtypeAnyInt(obj)) {
      if (!checkNumpyDtypeUint32(obj)) {
        printCopyWarning("labels", getDtype(obj), "uint32");
      }
      auto labels = obj.cast<NumpyArray<uint32_t>>();
      return BoltDatasetNumpyContext(labels, batch_size);
    }

    if (!checkNumpyDtypeFloat32(obj)) {
      printCopyWarning("data", getDtype(obj), "float32");
    }

    NumpyArray<float> data = obj.cast<NumpyArray<float>>();
    uint32_t input_dim = data.ndim() == 1 ? 1 : data.shape(1);
    if (input_dim != getInputDim()) {
      throw std::invalid_argument("Cannot pass array with input dimension " +
                                  std::to_string(input_dim) +
                                  " to network with input dim " +
                                  std::to_string(getInputDim()));
    }

    return BoltDatasetNumpyContext(data, batch_size);
  }

  BoltDatasetNumpyContext convertPyObjectToBoltDataset(const py::object& obj,
                                                       uint32_t batch_size,
                                                       bool is_labels) {
    if (isBoltDataset(obj)) {
      return BoltDatasetNumpyContext(obj.cast<dataset::BoltDatasetPtr>());
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

    bool alloc_success =
        allocateActivations(num_samples, inference_output_dim, &active_neurons,
                            &activations, output_sparse);

    auto metric_data =
        DLRM::predict(test_data, test_labels, active_neurons, activations,
                      use_sparse_inference, metrics, verbose, batch_limit);
    py::dict py_metric_data = py::cast(metric_data);

    return constructNumpyArrays(std::move(py_metric_data), num_samples,
                                inference_output_dim, active_neurons,
                                activations, output_sparse, alloc_success);
  }
};

}  // namespace thirdai::bolt::python

CEREAL_REGISTER_TYPE(thirdai::bolt::python::PyNetwork)
