#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/DLRM.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <pybind11/cast.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <fstream>
#include <iostream>
#include <limits>
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

  MetricData train(
      dataset::InMemoryDataset<dataset::BoltInputBatch>& train_data,
      // Clang tidy is disabled for this line because it wants to pass by
      // reference, but shared_ptrs should not be passed by reference
      const LossFunction& loss_fn,  // NOLINT
      float learning_rate, uint32_t epochs, uint32_t rehash, uint32_t rebuild,
      const std::vector<std::string>& metric_names, bool verbose) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));
    return FullyConnectedNetwork::train(train_data, loss_fn, learning_rate, epochs,
                                        1, rehash, rebuild, metric_names,
                                        verbose);
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
    auto data = thirdai::dataset::python::denseBoltDatasetFromNumpy(
        examples, labels, batch_size);

    // Prent clang tidy because it wants to pass the smart pointer by reference
    return train(data, loss_fn, learning_rate, epochs, rehash,  // NOLINT
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
    auto data = thirdai::dataset::python::sparseBoltDatasetFromNumpy(
        x_idxs, x_vals, x_offsets, y_idxs, y_vals, y_offsets, batch_size);

    // Prent clang tidy because it wants to pass the smart pointer by reference
    return train(data, loss_fn, learning_rate, epochs, rehash,  // NOLINT
                 rebuild, metrics, verbose);
  }

  std::pair<MetricData, py::object> predict(
      const dataset::InMemoryDataset<dataset::BoltInputBatch>& test_data,
      const std::vector<std::string>& metrics = {}, bool verbose = true,
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max()) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));

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

  std::pair<MetricData,
            py::array_t<float, py::array::c_style | py::array::forcecast>>
  predictWithDenseNumpyArray(
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          examples,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          labels,
      uint32_t batch_size, const std::vector<std::string>& metrics,
      bool verbose, uint32_t batch_limit) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));

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

  std::pair<MetricData,
            py::array_t<float, py::array::c_style | py::array::forcecast>>
  predictWithSparseNumpyArray(
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
    auto data = thirdai::dataset::python::sparseBoltDatasetFromNumpy(
        x_idxs, x_vals, x_offsets, y_idxs, y_vals, y_offsets, batch_size);
    uint32_t num_samples = x_offsets.shape()[0] - 1;
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

CEREAL_REGISTER_TYPE(thirdai::bolt::python::PyNetwork)
