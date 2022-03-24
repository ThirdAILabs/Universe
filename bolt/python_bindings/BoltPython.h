#pragma once

#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/networks/DLRM.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

void createBoltSubmodule(py::module_& module);

class PyNetwork final : public FullyConnectedNetwork {
 public:
  PyNetwork(std::vector<bolt::FullyConnectedLayerConfig> configs,
            uint64_t input_dim)
      : FullyConnectedNetwork(std::move(configs), input_dim) {}

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

  template <typename BATCH_T>
  std::vector<int64_t> trainWithPythonStdout(
    const dataset::InMemoryDataset<BATCH_T>& train_data, float learning_rate,
    uint32_t epochs, uint32_t rehash_in, uint32_t rebuild_in) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
      std::cout,
      py::module_::import("sys").attr("stdout")
    );
    return train(train_data, learning_rate, epochs, rehash_in, rebuild_in);
  }

  template <typename BATCH_T>
  float predictWithPythonStdout(
    const dataset::InMemoryDataset<BATCH_T>& test_data, uint32_t batch_limit) {
    // Redirect to python output.
    py::scoped_ostream_redirect stream(
      std::cout,
      py::module_::import("sys").attr("stdout")
    );
    return predict(test_data, batch_limit);
  }

  // Does not return py::array_t because this is consistent with the original
  // train method.
  std::vector<int64_t> trainWithDenseNumpyArray(
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          examples,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          labels,
      uint32_t batch_size, float learning_rate, uint32_t epochs,
      uint32_t rehash, uint32_t rebuild) {
    py::scoped_ostream_redirect stream(
      std::cout,
      py::module_::import("sys").attr("stdout")
    );
    uint32_t starting_id = 0;
    auto data = thirdai::dataset::python::denseInMemoryDatasetFromNumpy(
        examples, labels, batch_size, starting_id);
    return train(data, learning_rate, epochs, rehash, rebuild);
  }

  float predictWithDenseNumpyArray(
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          examples,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          labels,
      uint32_t batch_size, uint32_t batch_limit) {
    py::scoped_ostream_redirect stream(
      std::cout,
      py::module_::import("sys").attr("stdout")
    );
    uint32_t starting_id = 0;
    auto data = thirdai::dataset::python::denseInMemoryDatasetFromNumpy(
        examples, labels, batch_size, starting_id);
    return predict(data, batch_limit);
  }
  
  std::vector<int64_t> trainWithSparseNumpyArray(
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          x_idxs,
	  const py::array_t<float, py::array::c_style | py::array::forcecast>&
          x_vals,
	  const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          x_offsets,
	  const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          y_idxs,
	  const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          y_offsets,
      uint32_t batch_size, float learning_rate, uint32_t epochs,
      uint32_t rehash, uint32_t rebuild) {
    py::scoped_ostream_redirect stream(
      std::cout,
      py::module_::import("sys").attr("stdout")
    );
    uint32_t starting_id = 0;
    auto data = thirdai::dataset::python::sparseInMemoryDatasetFromNumpy(
        x_idxs, x_vals, x_offsets, y_idxs, y_offsets, batch_size, starting_id);
    return train(data, learning_rate, epochs, rehash, rebuild);
  }

  float predictWithSparseNumpyArray(
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          x_idxs,
	  const py::array_t<float, py::array::c_style | py::array::forcecast>&
          x_vals,
	  const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          x_offsets,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          y_idxs,
	  const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
          y_offsets,
      uint32_t batch_size, uint32_t batch_limit) {
    py::scoped_ostream_redirect stream(
      std::cout,
      py::module_::import("sys").attr("stdout")
    );
    uint32_t starting_id = 0;
    auto data = thirdai::dataset::python::sparseInMemoryDatasetFromNumpy(
        x_idxs, x_vals, x_offsets, y_idxs, y_offsets, batch_size, starting_id);
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
    float* scores = new float[test_data.len() * _output_dim];

    DLRM::predict(test_data, scores);

    py::capsule free_when_done(
        scores, [](void* ptr) { delete static_cast<float*>(ptr); });

    return py::array_t<float>({static_cast<size_t>(test_data.len()),
                               static_cast<size_t>(_output_dim)},
                              {_output_dim * sizeof(float), sizeof(float)},
                              scores, free_when_done);
  }
};

}  // namespace thirdai::bolt::python