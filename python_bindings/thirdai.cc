#include "../bolt/src/Network.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace thirdai::bolt;

namespace thirdai::python {

class PyNetwork final : public Network {
 public:
  PyNetwork(std::vector<bolt::LayerConfig> configs, uint64_t input_dim)
      : Network(std::move(configs), input_dim) {}

  py::array_t<float> GetWeightMatrix(uint32_t layer_index) {
    if (layer_index >= num_layers) {
      return py::none();
    }

    float* mem = layers[layer_index]->GetWeights();

    py::capsule free_when_done(mem, [](void* ptr) { delete (float*)ptr; });

    size_t dim = configs[layer_index].dim;
    size_t prev_dim =
        (layer_index > 0) ? configs[layer_index - 1].dim : input_dim;

    return py::array_t<float>({dim, prev_dim},
                              {prev_dim * sizeof(float), sizeof(float)}, mem,
                              free_when_done);
  }

  py::array_t<float> GetBiasVector(uint32_t layer_index) {
    if (layer_index >= num_layers) {
      return py::none();
    }

    float* mem = layers[layer_index]->GetBiases();

    py::capsule free_when_done(mem, [](void* ptr) { delete (float*)ptr; });

    size_t dim = configs[layer_index].dim;

    return py::array_t<float>({dim}, {sizeof(float)}, mem, free_when_done);
  }
};

}  // namespace thirdai::python

PYBIND11_MODULE(thirdai, m) {
  auto submodule = m.def_submodule("bolt");

  py::class_<thirdai::bolt::SamplingConfig>(submodule, "SamplingConfig")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(), py::arg("hashes_per_table"),
           py::arg("num_tables"), py::arg("range_pow"), py::arg("reservoir_size"))
      .def(py::init<>());

  py::class_<thirdai::bolt::LayerConfig>(submodule, "LayerConfig")
      .def(py::init<uint64_t, float, std::string,
                    thirdai::bolt::SamplingConfig>(),
           py::arg("dim"), py::arg("load_factor"),
           py::arg("activation_function"), py::arg("sampling_config"))
      .def(py::init<uint64_t, std::string>(), py::arg("dim"),
           py::arg("activation_function"))
      .def(py::init<uint64_t, float, std::string>(), py::arg("dim"),
           py::arg("load_factor"), py::arg("activation_function"));

  py::class_<thirdai::python::PyNetwork>(submodule, "Network")
      .def(py::init<std::vector<thirdai::bolt::LayerConfig>, uint64_t>(),
           py::arg("layers"), py::arg("input_dim"))
      .def("Train", &thirdai::python::PyNetwork::Train, py::arg("batch_size"),
           py::arg("train_data"), py::arg("test_data"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("rehash") = 0,
           py::arg("rebuild") = 0, py::arg("max_test_batches") = 0)
      .def("GetWeightMatrix", &thirdai::python::PyNetwork::GetWeightMatrix,
           py::arg("layer_index"))
      .def("GetBiasVector", &thirdai::python::PyNetwork::GetBiasVector,
           py::arg("layer_index"))
      .def("GetNumLayers", &thirdai::python::PyNetwork::GetNumLayers)
      .def("GetLayerSizes", &thirdai::python::PyNetwork::GetLayerSizes)
      .def("GetInputDim", &thirdai::python::PyNetwork::GetInputDim)
      .def("GetActivationFunctions",
           &thirdai::python::PyNetwork::GetActivationFunctions);
}