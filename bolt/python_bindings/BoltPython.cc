#include "BoltPython.h"

namespace thirdai::bolt::python {

void createBoltSubmodule(py::module_& module) {
  auto bolt_submodule = module.def_submodule("bolt");

  py::class_<thirdai::bolt::SamplingConfig>(bolt_submodule, "SamplingConfig")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
           py::arg("hashes_per_table"), py::arg("num_tables"),
           py::arg("range_pow"), py::arg("reservoir_size"))
      .def(py::init<>());

  py::class_<thirdai::bolt::FullyConnectedLayerConfig>(bolt_submodule,
                                                       "LayerConfig")
      .def(py::init<uint64_t, float, std::string,
                    thirdai::bolt::SamplingConfig>(),
           py::arg("dim"), py::arg("load_factor"),
           py::arg("activation_function"), py::arg("sampling_config"))
      .def(py::init<uint64_t, std::string>(), py::arg("dim"),
           py::arg("activation_function"))
      .def(py::init<uint64_t, float, std::string>(), py::arg("dim"),
           py::arg("load_factor"), py::arg("activation_function"));

  py::class_<thirdai::bolt::EmbeddingLayerConfig>(bolt_submodule,
                                                  "EmbeddingLayerConfig")
      .def(py::init<uint32_t, uint32_t, uint32_t>(),
           py::arg("num_embedding_lookups"), py::arg("lookup_size"),
           py::arg("log_embedding_block_size"));

  py::class_<PyNetwork>(bolt_submodule, "Network")
      .def(py::init<std::vector<thirdai::bolt::FullyConnectedLayerConfig>,
                    uint64_t>(),
           py::arg("layers"), py::arg("input_dim"))
      .def("Train", &PyNetwork::train<thirdai::dataset::SparseBatch>, py::arg("train_data"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("rehash") = 0,
           py::arg("rebuild") = 0)
      .def("Train", &PyNetwork::train<thirdai::dataset::DenseBatch>, py::arg("train_data"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("rehash") = 0,
           py::arg("rebuild") = 0)
      .def("Test", &PyNetwork::test<thirdai::dataset::SparseBatch>, py::arg("test_data"),
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max())
      .def("Test", &PyNetwork::test<thirdai::dataset::DenseBatch>, py::arg("test_data"),
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max())
      .def("UseSparseInference", &PyNetwork::useSparseInference)
      .def("GetWeightMatrix", &PyNetwork::getWeightMatrix,
           py::arg("layer_index"))
      .def("GetBiasVector", &PyNetwork::getBiasVector, py::arg("layer_index"))
      .def("GetNumLayers", &PyNetwork::getNumLayers)
      .def("GetLayerSizes", &PyNetwork::getLayerSizes)
      .def("GetInputDim", &PyNetwork::getInputDim)
      .def("GetActivationFunctions", &PyNetwork::getActivationFunctions);

  py::class_<PyDLRM>(bolt_submodule, "DLRM")
      .def(py::init<thirdai::bolt::EmbeddingLayerConfig,
                    std::vector<thirdai::bolt::FullyConnectedLayerConfig>,
                    std::vector<thirdai::bolt::FullyConnectedLayerConfig>,
                    uint32_t>(),
           py::arg("embedding_layer"), py::arg("bottom_mlp"),
           py::arg("top_mlp"), py::arg("input_dim"))
      .def("Train", &PyDLRM::train, py::arg("train_data"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("rehash"),
           py::arg("rebuild"))
      .def("Test", &PyDLRM::test, py::arg("test_data"));
}

}  // namespace thirdai::bolt::python
