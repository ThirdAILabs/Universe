#include "../bolt/src/Network.h"
#include "../flash/src/Flash.h"
#include "../utils/dataset/Dataset.h"
#include "../utils/dataset/svm/SVMDataset.h"
#include "../utils/hashing/DensifiedMinHash.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <omp.h>

namespace py = pybind11;

using thirdai::bolt::Network;

namespace thirdai::python {

class PyNetwork final : public Network {
 public:
  PyNetwork(std::vector<bolt::LayerConfig> configs, uint64_t input_dim)
      : Network(std::move(configs), input_dim) {}

  py::array_t<float> GetWeightMatrix(uint32_t layer_index) {
    if (layer_index >= _num_layers) {
      return py::none();
    }

    float* mem = _layers[layer_index]->GetWeights();

    py::capsule free_when_done(
        mem, [](void* ptr) { delete static_cast<float*>(ptr); });

    size_t dim = _configs[layer_index].dim;
    size_t prev_dim =
        (layer_index > 0) ? _configs[layer_index - 1].dim : _input_dim;

    return py::array_t<float>({dim, prev_dim},
                              {prev_dim * sizeof(float), sizeof(float)}, mem,
                              free_when_done);
  }

  py::array_t<float> GetBiasVector(uint32_t layer_index) {
    if (layer_index >= _num_layers) {
      return py::none();
    }

    float* mem = _layers[layer_index]->GetBiases();

    py::capsule free_when_done(
        mem, [](void* ptr) { delete static_cast<float*>(ptr); });

    size_t dim = _configs[layer_index].dim;

    return py::array_t<float>({dim}, {sizeof(float)}, mem, free_when_done);
  }
};

// TODO(any): Move this when we refactor the dataset class
utils::Dataset* createSVMDataset(const std::string& dataset_path,
                                 uint64_t batch_size,
                                 uint64_t batches_per_load) {
  return new utils::SVMDataset(dataset_path, batch_size, batches_per_load);
}

}  // namespace thirdai::python

// TODO(all): Figure out naming convention for python exposed classes and
// methods
// TODO(any): Add docstrings to methods
PYBIND11_MODULE(thirdai, m) {  // NOLINT

  auto utils_submodule = m.def_submodule("utils");

  py::class_<thirdai::utils::Batch>(utils_submodule, "Batch");  // NOLINT

  py::class_<thirdai::utils::Dataset>(utils_submodule, "Dataset")
      .def("__getitem__", &thirdai::utils::Dataset::operator[],
           py::return_value_policy::reference)
      .def("LoadNextSetOfBatches", &thirdai::utils::Dataset::loadNextBatchSet);

  py::class_<thirdai::utils::HashFunction>(utils_submodule, "HashFunction")
      .def("GetNumTables", &thirdai::utils::HashFunction::numTables)
      .def("GetRange", &thirdai::utils::HashFunction::range);

  py::class_<thirdai::utils::DensifiedMinHash, thirdai::utils::HashFunction>(
      utils_submodule, "DensifiedMinHash")
      .def(py::init<uint32_t, uint32_t, uint32_t>(),
           py::arg("hashes_per_table"), py::arg("num_tables"),
           py::arg("range"));

  utils_submodule.def("load_svm", &thirdai::python::createSVMDataset,
                      py::arg("dataset_path"), py::arg("batch_size"),
                      py::arg("batches_per_load"),
                      "Load an SVM dataset from memory, ready for use with "
                      "e.g. BOLT or FLASH.");

  utils_submodule.def("set_global_num_threads", &omp_set_num_threads,
                      py::arg("max_num_threads"));

  // TODO(any): Rename from flash, and fix all other places in the code
  auto flash_submodule = m.def_submodule("search");
  py::class_<thirdai::search::Flash<uint64_t>>(flash_submodule, "Flash")
      .def(py::init<const thirdai::utils::HashFunction&>(),
           py::arg("hash_function"))
      .def("AddDataset", &thirdai::search::Flash<uint64_t>::addDataset,
           py::arg("dataset"))
      .def("QueryBatch", &thirdai::search::Flash<uint64_t>::queryBatch,
           py::arg("batch"), py::arg("top_k"), py::arg("pad_zeros") = false);

  auto bolt_submodule = m.def_submodule("bolt");

  py::class_<thirdai::bolt::SamplingConfig>(bolt_submodule, "SamplingConfig")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
           py::arg("hashes_per_table"), py::arg("num_tables"),
           py::arg("range_pow"), py::arg("reservoir_size"))
      .def(py::init<>());

  py::class_<thirdai::bolt::LayerConfig>(bolt_submodule, "LayerConfig")
      .def(py::init<uint64_t, float, std::string,
                    thirdai::bolt::SamplingConfig>(),
           py::arg("dim"), py::arg("load_factor"),
           py::arg("activation_function"), py::arg("sampling_config"))
      .def(py::init<uint64_t, std::string>(), py::arg("dim"),
           py::arg("activation_function"))
      .def(py::init<uint64_t, float, std::string>(), py::arg("dim"),
           py::arg("load_factor"), py::arg("activation_function"));

  py::class_<thirdai::python::PyNetwork>(bolt_submodule, "Network")
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
           &thirdai::python::PyNetwork::GetActivationFunctions)
      .def("GetAccuracyPerEpoch",
           &thirdai::python::PyNetwork::GetAccuracyPerEpoch)
      .def("GetTimePerEpoch", &thirdai::python::PyNetwork::GetTimePerEpoch)
      .def("GetFinalTestAccuracy",
           &thirdai::python::PyNetwork::GetFinalTestAccuracy);
}