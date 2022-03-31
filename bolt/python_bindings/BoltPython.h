#pragma once

#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/DLRM.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <cstddef>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace thirdai::bolt::python {

void createBoltSubmodule(py::module_& module);

class PyNetwork final : public FullyConnectedNetwork {
 public:
  PyNetwork(std::vector<bolt::FullyConnectedLayerConfig> configs,
            uint64_t input_dim)
      : FullyConnectedNetwork(std::move(configs), input_dim) {}

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
    auto data = thirdai::dataset::python::denseBoltDatasetFromNumpy(
        examples, labels, batch_size);

    // Prent clang tidy because it wants to pass the smart pointer by reference
    return train(data, loss_fn, learning_rate, epochs, rehash,  // NOLINT
                 rebuild, metrics, verbose);
  }

  std::pair<MetricData, py::object> predict(
      const dataset::InMemoryDataset<dataset::BoltInputBatch>& test_data,
      const std::vector<std::string>& metrics = {}, bool verbose = true,
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max()) {
    uint32_t num_samples = test_data.len();

    float* activations;
    try {
      activations = new float[num_samples * outputDim()];
    } catch (std::bad_alloc& e) {
      activations = nullptr;
      std::cout << "Out of memory error: cannot allocate " << num_samples
                << " x " << outputDim() << " array for activations"
                << std::endl;
    }

    auto metric_data = FullyConnectedNetwork::predict(
        test_data, activations, metrics, verbose, batch_limit);

    if (activations == nullptr) {
      return {metric_data, py::none()};
    }

    py::capsule free_when_done(
        activations, [](void* ptr) { delete static_cast<float*>(ptr); });

    py::array_t<float> activations_array(
        {num_samples, outputDim()},
        {outputDim() * sizeof(float), sizeof(float)}, activations,
        free_when_done);

    return {metric_data, activations_array};
  }

  py::object getEmbeddings(
      uint32_t layer_no, 
      const dataset::InMemoryDataset<dataset::BoltInputBatch>& test_data,
      uint32_t test_batch_size) {
    auto output = FullyConnectedNetwork::getEmbeddings(layer_no, test_data, test_batch_size);
    return py::cast(output);
  }

  py::object getEmbeddingsWithDenseNumpyArray(
      uint32_t layer_no, 
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          examples,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          labels,
      uint32_t test_batch_size) {
    
    //std::cout << "Construct dataset" << std::endl;
    auto test_data = thirdai::dataset::python::denseBoltDatasetFromNumpy(
        examples, labels, test_batch_size);

    auto output = FullyConnectedNetwork::getEmbeddings(layer_no, test_data, test_batch_size);
    std::cout << "return embeddings" << std::endl;
    return py::cast(output);
  }

  std::pair<MetricData,
            py::array_t<float, py::array::c_style | py::array::forcecast>>
  predictWithDenseNumpyArray(
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          examples,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          labels,
      uint32_t batch_size, const std::vector<std::string>& metrics,
      bool verbose, uint32_t batch_limit) {
    auto data = thirdai::dataset::python::denseBoltDatasetFromNumpy(
        examples, labels, batch_size);

    uint32_t num_samples = examples.shape()[0];
    float* activations;
    try {
      activations = new float[num_samples * outputDim()];
    } catch (std::bad_alloc& e) {
      activations = nullptr;
      std::cout << "Out of memory error: cannot allocate " << num_samples
                << " x " << outputDim() << " array for activations"
                << std::endl;
    }

    auto metric_data = FullyConnectedNetwork::predict(
        data, activations, metrics, verbose, batch_limit);

    if (activations == nullptr) {
      return {metric_data, py::none()};
    }

    py::capsule free_when_done(
        activations, [](void* ptr) { delete static_cast<float*>(ptr); });

    py::array_t<float> activations_array(
        {num_samples, outputDim()},
        {outputDim() * sizeof(float), sizeof(float)}, activations,
        free_when_done);

    return {metric_data, activations_array};
  }
};

class PyDLRM final : public DLRM {
 public:
  PyDLRM(bolt::EmbeddingLayerConfig embedding_config,
         std::vector<bolt::FullyConnectedLayerConfig> bottom_mlp_configs,
         std::vector<bolt::FullyConnectedLayerConfig> top_mlp_configs,
         uint32_t input_dim)
      : DLRM(embedding_config, std::move(bottom_mlp_configs),
             std::move(top_mlp_configs), input_dim) {}

  std::pair<MetricData,
            py::array_t<float, py::array::c_style | py::array::forcecast>>
  predict(const dataset::InMemoryDataset<dataset::ClickThroughBatch>& test_data,
          const std::vector<std::string>& metrics = {}, bool verbose = true,
          uint32_t batch_limit = std::numeric_limits<uint32_t>::max()) {
    uint32_t num_samples = test_data.len();
    float* activations;
    try {
      activations = new float[num_samples * outputDim()];
    } catch (std::bad_alloc& e) {
      activations = nullptr;
      std::cout << "Out of memory error: cannot allocate " << num_samples
                << " x " << outputDim() << " array for activations"
                << std::endl;
    }

    auto metric_data =
        DLRM::predict(test_data, activations, metrics, verbose, batch_limit);

    if (activations == nullptr) {
      return {metric_data, py::none()};
    }

    py::capsule free_when_done(
        activations, [](void* ptr) { delete static_cast<float*>(ptr); });

    py::array_t<float> activations_array(
        {num_samples, outputDim()},
        {outputDim() * sizeof(float), sizeof(float)}, activations,
        free_when_done);

    return {metric_data, activations_array};
  }
};

}  // namespace thirdai::bolt::python
