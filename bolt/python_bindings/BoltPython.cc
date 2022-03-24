#include "BoltPython.h"
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <pybind11/pybind11.h>

namespace thirdai::bolt::python {

void createBoltSubmodule(py::module_& module) {
  auto bolt_submodule = module.def_submodule("bolt");

  py::class_<thirdai::bolt::SamplingConfig>(bolt_submodule, "SamplingConfig")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
           py::arg("hashes_per_table"), py::arg("num_tables"),
           py::arg("range_pow"), py::arg("reservoir_size"))
      .def(py::init<>());

  py::enum_<ActivationFunction>(bolt_submodule, "ActivationFunctions")
      .value("ReLU", ActivationFunction::ReLU)
      .value("Linear", ActivationFunction::Linear)
      .value("Softmax", ActivationFunction::Softmax);

  bolt_submodule.def("getActivationFunction", &getActivationFunction,
                     py::arg("name"));

  py::class_<LossFunction>(bolt_submodule, "LossFunction");  // NOLINT

  py::class_<CategoricalCrossEntropyLoss, LossFunction>(
      bolt_submodule, "CategoricalCrossEntropyLoss")
      .def(py::init<>());

  py::class_<MeanSquaredError, LossFunction>(bolt_submodule, "MeanSquaredError")
      .def(py::init<>());

  py::class_<thirdai::bolt::FullyConnectedLayerConfig>(bolt_submodule,
                                                       "LayerConfig")
      .def(py::init<uint64_t, float, ActivationFunction,
                    thirdai::bolt::SamplingConfig>(),
           py::arg("dim"), py::arg("load_factor"),
           py::arg("activation_function"), py::arg("sampling_config"))
      .def(py::init<uint64_t, ActivationFunction>(), py::arg("dim"),
           py::arg("activation_function"))
      .def(py::init<uint64_t, float, ActivationFunction>(), py::arg("dim"),
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
      .def("train", &PyNetwork::train, py::arg("train_data"),
           py::arg("loss_fn"), py::arg("learning_rate"), py::arg("epochs"),
           py::arg("rehash") = 0, py::arg("rebuild") = 0,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true)
      .def("train", &PyNetwork::trainWithDenseNumpyArray,
           py::arg("train_examples"), py::arg("train_labels"),
           py::arg("batch_size"), py::arg("loss_fn"), py::arg("learning_rate"),
           py::arg("epochs"), py::arg("rehash") = 0, py::arg("rebuild") = 0,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true)
      .def("predict", &PyNetwork::predict, py::arg("test_data"),
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max())
      .def("predict", &PyNetwork::predictWithDenseNumpyArray,
           py::arg("test_examples"), py::arg("test_labels"),
           py::arg("batch_size"),
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max())
      .def("enable_sparse_inference", &PyNetwork::enableSparseInference);

  py::class_<PyDLRM>(bolt_submodule, "DLRM")
      .def(py::init<thirdai::bolt::EmbeddingLayerConfig,
                    std::vector<thirdai::bolt::FullyConnectedLayerConfig>,
                    std::vector<thirdai::bolt::FullyConnectedLayerConfig>,
                    uint32_t>(),
           py::arg("embedding_layer"), py::arg("bottom_mlp"),
           py::arg("top_mlp"), py::arg("input_dim"))
      .def("train", &PyDLRM::train, py::arg("train_data"), py::arg("loss_fn"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("rehash") = 0,
           py::arg("rebuild") = 0,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true)
      .def("predict", &PyDLRM::predict, py::arg("test_data"),
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max());
}

}  // namespace thirdai::bolt::python
