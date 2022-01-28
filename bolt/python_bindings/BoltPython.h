#pragma once

#include <bolt/layers/LayerConfig.h>
#include <bolt/networks/DLRM.h>
#include <bolt/networks/Network.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

void createBoltSubmodule(py::module_& module);

class PyNetwork final : public Network {
 public:
  PyNetwork(std::vector<bolt::FullyConnectedLayerConfig> configs,
            uint64_t input_dim)
      : Network(std::move(configs), input_dim) {}

  py::array_t<float> getWeightMatrix(uint32_t layer_index) {
    if (layer_index >= _num_layers) {
      return py::none();
    }

    float* mem = _layers[layer_index]->getWeights();

    py::capsule free_when_done(
        mem, [](void* ptr) { delete static_cast<float*>(ptr); });

    size_t dim = _configs[layer_index].dim;
    size_t prev_dim =
        (layer_index > 0) ? _configs[layer_index - 1].dim : _input_dim;

    return py::array_t<float>({dim, prev_dim},
                              {prev_dim * sizeof(float), sizeof(float)}, mem,
                              free_when_done);
  }

  py::array_t<float> getBiasVector(uint32_t layer_index) {
    if (layer_index >= _num_layers) {
      return py::none();
    }

    float* mem = _layers[layer_index]->getBiases();

    py::capsule free_when_done(
        mem, [](void* ptr) { delete static_cast<float*>(ptr); });

    size_t dim = _configs[layer_index].dim;

    return py::array_t<float>({dim}, {sizeof(float)}, mem, free_when_done);
  }

  // Does not return py::array_t because this is consistent with the original
  // train method.
  std::vector<int64_t> trainWithDenseNumpyArray(
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          examples,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          labels,
      uint32_t batch_size, float learning_rate, uint32_t epochs,
      uint64_t starting_id, uint32_t rehash, uint32_t rebuild) {
    auto data = thirdai::dataset::python::denseInMemoryDatasetFromNumpy(
        examples, labels, batch_size, starting_id);
    return train(data, learning_rate, epochs, rehash, rebuild);
  }

  float testWithDenseNumpyArray(
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          examples,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          labels,
      uint32_t batch_size, uint64_t starting_id, uint32_t batch_limit) {
    auto data = thirdai::dataset::python::denseInMemoryDatasetFromNumpy(
        examples, labels, batch_size, starting_id);
    return predict(data, batch_limit);
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

  py::array_t<float> predict(
      const dataset::InMemoryDataset<dataset::ClickThroughBatch>& test_data) {
    py::array_t<float> scores({static_cast<uint32_t>(test_data.len())});

    DLRM::predict(test_data, scores.mutable_data());

    return scores;
  }
};

}  // namespace thirdai::bolt::python