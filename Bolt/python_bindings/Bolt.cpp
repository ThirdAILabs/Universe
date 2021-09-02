#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define PYTHON_BINDINGS

#include "../src/Network.h"

namespace py = pybind11;

PYBIND11_MODULE(thirdai, m) {
  auto submodule = m.def_submodule("bolt");

  py::class_<bolt::SamplingConfig>(submodule, "SamplingConfig")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(), py::arg("K"), py::arg("L"),
           py::arg("range_pow"), py::arg("reservoir_size"))
      .def(py::init<>());

  py::class_<bolt::LayerConfig>(submodule, "LayerConfig")
      .def(py::init<uint64_t, float, std::string, bolt::SamplingConfig>(), py::arg("dim"),
           py::arg("load_factor"), py::arg("activation_function"), py::arg("sampling_config"))
      .def(py::init<uint64_t, std::string>(), py::arg("dim"), py::arg("activation_function"))
      .def(py::init<uint64_t, float, std::string>(), py::arg("dim"), py::arg("load_factor"),
           py::arg("activation_function"));

  py::class_<bolt::Network>(submodule, "Network")
      .def(py::init<std::vector<bolt::LayerConfig>, uint64_t>(), py::arg("layers"),
           py::arg("input_dim"))
      .def("Train", &bolt::Network::Train, py::arg("batch_size"), py::arg("train_data"),
           py::arg("test_data"), py::arg("learning_rate"), py::arg("epochs"), py::arg("rehash") = 0,
           py::arg("rebuild") = 0, py::arg("max_test_batches") = 0)
      .def("GetWeightMatrix", &bolt::Network::GetWeightMatrix, py::arg("layer_index"))
      .def("GetBiasVector", &bolt::Network::GetBiasVector, py::arg("layer_index"));
}