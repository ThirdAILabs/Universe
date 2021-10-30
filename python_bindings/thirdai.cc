#include "../bolt/networks/Network.h"
#include "../flash/src/Flash.h"
#include "../utils/dataset/Dataset.h"
#include "../utils/dataset/batch_types/SparseBatch.h"
#include "../utils/hashing/DensifiedMinHash.h"
#include "../utils/hashing/FastSRP.h"
#include <_types/_uint32_t.h>
#include <_types/_uint64_t.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <string>
#include <vector>
#ifndef __clang__
#include <omp.h>
#endif
#include <stdexcept>

namespace py = pybind11;

using thirdai::bolt::Network;

using thirdai::utils::DenseBatch;
using thirdai::utils::DenseVector;
using thirdai::utils::InMemoryDataset;
using thirdai::utils::SparseBatch;
using thirdai::utils::SparseVector;
using thirdai::utils::StreamedDataset;

using thirdai::utils::DensifiedMinHash;
using thirdai::utils::FastSRP;
using thirdai::utils::HashFunction;

using Flash64 = thirdai::search::Flash<uint64_t>;

namespace thirdai::python {

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
};

// TODO(josh): Is this method in a good place?
// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html?highlight=numpy#arrays
// for explanation of why we do py::array::c_style and py::array::forcecase.
// Ensures array is an array of floats in dense row major order.
static DenseBatch creatNumpyBatch(
    const pybind11::array_t<float, pybind11::array::c_style |
                                       pybind11::array::forcecast>& data,
    uint64_t starting_id) {
  const pybind11::buffer_info data_buf = data.request();
  const auto shape = data_buf.shape;
  if (shape.size() != 2) {
    throw std::invalid_argument(
        "For now, Numpy datasets must be 2D (each row is a dense data "
        "vectors).");
  }

  uint64_t num_vectors = static_cast<uint64_t>(shape.at(0));
  uint64_t dimension = static_cast<uint64_t>(shape.at(1));
  float* raw_data = static_cast<float*>(data_buf.ptr);
  std::vector<utils::DenseVector> batch_vectors;
  for (uint64_t vec_id = 0; vec_id < num_vectors; vec_id++) {
    // owns_data = false because we don't want the numpy array to be deleted
    // if this batch (and thus the underlying vectors) get deleted
    bool owns_data = false;
    batch_vectors.emplace_back(dimension, raw_data + dimension * vec_id,
                               owns_data);
  }

  return DenseBatch(std::move(batch_vectors), starting_id);
}

}  // namespace thirdai::python

// TODO(all): Figure out naming convention for python exposed classes and
// methods
// TODO(any): Add docstrings to methods
// TODO(any): Can we remove redudancy in the bindings?
PYBIND11_MODULE(thirdai, m) {  // NOLINT

  auto utils_submodule = m.def_submodule("utils");

  // TODO(josh): We can expose dense batches and dense/sparse vectors and
  // let them be cast directly to numpy arrays, see
  // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html

  py::class_<SparseVector>(utils_submodule, "SparseVector")
      .def("get_length", &SparseVector::length)
      .def("__getitem__", &SparseVector::at);

  py::class_<DenseVector>(utils_submodule, "DenseVector")
      .def("get_dim", &DenseVector::dim)
      .def("__getitem__", &DenseVector::at);

  py::class_<DenseBatch>(utils_submodule, "DenseBatch")
      // This is reference_internal so that even if there is no reference
      // to the batch, but there is a reference to a vector returned
      // from this operator, the vector won't be deleted. Note we use the
      // memory safe at() since this is a python binding.
      .def("__getitem__", &DenseBatch::at,
           py::return_value_policy::reference_internal,
           "Returns the ith vector in this batch.")
      .def("get_batch_size", &DenseBatch::getBatchSize);

  py::class_<SparseBatch>(utils_submodule, "SparseBatch")
      .def("__getitem__", &SparseBatch::at,
           py::return_value_policy::reference_internal,
           "Returns the ith vector in this batch.")
      .def("get_batch_size", &SparseBatch::getBatchSize);

  py::class_<InMemoryDataset<DenseBatch>>(utils_submodule,  // NOLINT
                                          "InMemoryDenseDataset")
      .def("get_num_batches", &InMemoryDataset<DenseBatch>::numBatches,
           "Returns the number of stored batches.")
      // This is reference_internal so that even if there is no reference
      // to the outer dataset, but there is a reference to a batch returned
      // from this operator, the batch won't be deleted.
      .def("__getitem__", &InMemoryDataset<DenseBatch>::at,
           py::return_value_policy::reference_internal,
           "Returns the currently stored ith batch.");

  py::class_<InMemoryDataset<SparseBatch>>(utils_submodule,
                                           "InMemorySparseDataset")
      .def("get_num_batches", &InMemoryDataset<SparseBatch>::numBatches,
           "Returns the number of stored batches.")
      .def("__getitem__", &InMemoryDataset<SparseBatch>::at,
           py::return_value_policy::reference_internal,
           "Returns the currently stored ith batch.");

  utils_submodule.def(
      "loadInMemorySvmDataset",
      &InMemoryDataset<SparseBatch>::loadInMemorySvmDataset,
      py::arg("filename"), py::arg("batch_size") = 10000,
      "Constructs a sparse dataset from a given file with batch sizes of "
      "a given size, and attempts to read the"
      " entire file into memory.");

  utils_submodule.def("create_numpy_batch", &thirdai::python::creatNumpyBatch,
                      py::keep_alive<0, 1>(), py::arg("numpy_array"),
                      py::arg("starting_id") = 0);

  py::class_<HashFunction>(
      utils_submodule, "HashFunction",
      "Represents an abstract hash function that maps input DenseVectors and "
      "SparseVectors to sets of positive integers")
      .def("get_num_tables", &HashFunction::numTables,
           "Returns the number of hash tables in this hash function, which is "
           "equivalently the number of hashes that get returned by the "
           "function for each input.")
      .def("get_range", &HashFunction::range,
           "All hashes returned from this function will be >= 0 and <= "
           "get_range().")
      .def("hash_batch",
           static_cast<std::vector<uint32_t> (HashFunction::*)(
               const DenseBatch&) const>(&HashFunction::hashBatchParallel),
           py::arg("batch"),
           "Hashes a batch using this hash function, returning the results in "
           "a 1D vector in row major order")
      .def("hash_batch",
           static_cast<std::vector<uint32_t> (HashFunction::*)(
               const SparseBatch&) const>(&HashFunction::hashBatchParallel),
           py::arg("batch"),
           "Hashes a batch using this hash function, returning the results in "
           "a 1D vector in row major order");

  py::class_<DensifiedMinHash, HashFunction>(
      utils_submodule, "MinHash",
      "A concrete implementation of a HashFunction that performs an extremly "
      "efficient minhash. A statistical estimator of jaccard similarity.")
      .def(py::init<uint32_t, uint32_t, uint32_t>(),
           py::arg("hashes_per_table"), py::arg("num_tables"),
           py::arg("range"));

  py::class_<FastSRP, HashFunction>(
      utils_submodule, "SignedRandomProjection",
      "A concrete implementation of a HashFunction that performs an extremly "
      "efficient signed random projection. A statistical estimator of cossine "
      "similarity.")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
           py::arg("input_dim"), py::arg("hashes_per_table"),
           py::arg("num_tables"), py::arg("range") = UINT32_MAX);

#ifndef __clang__
  utils_submodule.def("set_global_num_threads", &omp_set_num_threads,
                      py::arg("max_num_threads"));
#endif

  // TODO(any): Rename from flash, and fix all other places in the code
  auto flash_submodule = m.def_submodule("search");
  py::class_<Flash64>(
      flash_submodule, "Flash",
      "Flash is an index for performing near neighbour search. To use it, "
      "construct an index by calling one or more of add_dataset or "
      "add_batch. You may need to stream or read in a Dataset from disk "
      "using one of our utility data wrappers.")
      .def(py::init<HashFunction&, uint32_t>(), py::arg("hash_function"),
           py::arg("reservoir_size"),
           "Build a Flash index where all hash "
           "buckets have a max size reservoir_size.")
      .def(py::init<HashFunction&>(), py::arg("hash_function"),
           "Build a Flash index where buckets do not have a max size.")
      // See https://github.com/pybind/pybind11/issues/1153 for why we can't
      // do a py::overload_cas and instead need to use a static cast instead
      .def("add_dataset",
           static_cast<void (Flash64::*)(InMemoryDataset<SparseBatch>&)>(
               &Flash64::addDataset),
           py::arg("dataset"))
      .def("add_dataset",
           static_cast<void (Flash64::*)(InMemoryDataset<DenseBatch>&)>(
               &Flash64::addDataset),
           py::arg("dataset"))
      .def("add_dataset",
           static_cast<void (Flash64::*)(StreamedDataset<SparseBatch>&)>(
               &Flash64::addDataset),
           py::arg("dataset"))
      .def("add_dataset",
           static_cast<void (Flash64::*)(StreamedDataset<DenseBatch>&)>(
               &Flash64::addDataset),
           py::arg("dataset"))
      .def("add_batch",
           static_cast<void (Flash64::*)(const SparseBatch&)>(
               &Flash64::addBatch),
           py::arg("batch"))
      .def(
          "add_batch",
          static_cast<void (Flash64::*)(const DenseBatch&)>(&Flash64::addBatch),
          py::arg("batch"))
      .def("query",
           static_cast<std::vector<std::vector<uint64_t>> (Flash64::*)(
               const SparseBatch&, uint32_t, bool) const>(&Flash64::queryBatch),
           py::arg("dense_batch"), py::arg("top_k"),
           py::arg("pad_zeros") = false)
      .def("query",
           static_cast<std::vector<std::vector<uint64_t>> (Flash64::*)(
               const DenseBatch&, uint32_t, bool) const>(&Flash64::queryBatch),
           py::arg("dense_batch"), py::arg("top_k"),
           py::arg("pad_zeros") = false);
  .def("query",
       static_cast<std::vector<uint64_t> (Flash64::*)(
           const DenseVector&, uint32_t, bool) const>(&Flash64::queryBatch),
       py::arg("dense_vector"), py::arg("top_k"), py::arg("pad_zeros") = false);

  auto bolt_submodule = m.def_submodule("bolt");

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

  py::class_<thirdai::python::PyNetwork>(bolt_submodule, "Network")
      .def(py::init<std::vector<thirdai::bolt::FullyConnectedLayerConfig>,
                    uint64_t>(),
           py::arg("layers"), py::arg("input_dim"))
      .def("Train", &thirdai::python::PyNetwork::train, py::arg("batch_size"),
           py::arg("train_data"), py::arg("test_data"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("rehash") = 0,
           py::arg("rebuild") = 0, py::arg("max_test_batches") = 0)
      .def("GetWeightMatrix", &thirdai::python::PyNetwork::getWeightMatrix,
           py::arg("layer_index"))
      .def("GetBiasVector", &thirdai::python::PyNetwork::getBiasVector,
           py::arg("layer_index"))
      .def("GetNumLayers", &thirdai::python::PyNetwork::getNumLayers)
      .def("GetLayerSizes", &thirdai::python::PyNetwork::getLayerSizes)
      .def("GetInputDim", &thirdai::python::PyNetwork::getInputDim)
      .def("GetActivationFunctions",
           &thirdai::python::PyNetwork::getActivationFunctions)
      .def("GetAccuracyPerEpoch",
           &thirdai::python::PyNetwork::getAccuracyPerEpoch)
      .def("GetTimePerEpoch", &thirdai::python::PyNetwork::getTimePerEpoch)
      .def("GetFinalTestAccuracy",
           &thirdai::python::PyNetwork::getFinalTestAccuracy);
}