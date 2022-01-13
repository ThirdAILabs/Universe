#include <bolt/networks/DLRM.h>
#include <bolt/networks/Network.h>
#include <hashing/src/DensifiedMinHash.h>
#include <hashing/src/FastSRP.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/batch_types/SparseBatch.h>
#include <flash/src/Flash.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
#include <string>
#include <vector>
#ifndef __clang__
#include <omp.h>
#endif
#include <stdexcept>

namespace py = pybind11;

using thirdai::bolt::DLRM;
using thirdai::bolt::Network;

using thirdai::dataset::DenseBatch;
using thirdai::dataset::DenseVector;
using thirdai::dataset::InMemoryDataset;
using thirdai::dataset::SparseBatch;
using thirdai::dataset::SparseVector;

using thirdai::hashing::DensifiedMinHash;
using thirdai::hashing::FastSRP;
using thirdai::hashing::HashFunction;

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

using ClickThroughDataset =
    thirdai::dataset::InMemoryDataset<thirdai::dataset::ClickThroughBatch>;

ClickThroughDataset loadClickThorughDataset(const std::string& filename,
                                            uint32_t batch_size,
                                            uint32_t num_dense_features,
                                            uint32_t num_categorical_features) {
  auto start = std::chrono::high_resolution_clock::now();
  thirdai::dataset::ClickThroughBatchFactory factory(num_dense_features,
                                                     num_categorical_features);
  ClickThroughDataset data(filename, batch_size, std::move(factory));
  auto end = std::chrono::high_resolution_clock::now();
  std::cout
      << "Read " << data.len() << " vectors in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;
  return data;
}

class PyDLRM final : public DLRM {
 public:
  PyDLRM(bolt::EmbeddingLayerConfig embedding_config,
         std::vector<bolt::FullyConnectedLayerConfig> bottom_mlp_configs,
         std::vector<bolt::FullyConnectedLayerConfig> top_mlp_configs,
         uint32_t input_dim)
      : DLRM(embedding_config, std::move(bottom_mlp_configs),
             std::move(top_mlp_configs), input_dim) {}

  py::array_t<float> test(
      const dataset::InMemoryDataset<dataset::ClickThroughBatch>& test_data) {
    py::array_t<float> scores({static_cast<uint32_t>(test_data.len())});

    testImpl(test_data, scores.mutable_data());

    return scores;
  }
};

// TODO(josh): Is this method in a good place?
// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html?highlight=numpy#arrays
// for explanation of why we do py::array::c_style and py::array::forcecase.
// Ensures array is an array of floats in dense row major order.
static SparseBatch wrapNumpyIntoSparseData(
    const std::vector<py::array_t<
        float, py::array::c_style | py::array::forcecast>>& sparse_values,
    const std::vector<
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>&
        sparse_indices,
    uint64_t starting_id) {
  if (sparse_values.size() != sparse_indices.size()) {
    throw std::invalid_argument(
        "Values and indices arrays must have the same number of elements.");
  }

  uint64_t num_vectors = sparse_values.size();

  std::vector<dataset::SparseVector> batch_vectors;
  for (uint64_t vec_id = 0; vec_id < num_vectors; vec_id++) {
    const py::buffer_info indices_buf = sparse_indices.at(vec_id).request();
    const py::buffer_info values_buf = sparse_values.at(vec_id).request();
    const auto indices_shape = indices_buf.shape;
    const auto values_shape = values_buf.shape;

    if (indices_shape.size() != 1 || values_shape.size() != 1) {
      throw std::invalid_argument(
          "For now, every entry in the indices and values arrays must be a 1D "
          "array.");
    }

    if (indices_shape.at(0) != values_shape.at(0)) {
      throw std::invalid_argument(
          "Corresponding indice and value entries must have the same number of "
          "values.");
    }

    bool owns_data = false;
    uint64_t length = indices_shape.at(0);
    batch_vectors.emplace_back(static_cast<uint32_t*>(indices_buf.ptr),
                               static_cast<float*>(values_buf.ptr), length,
                               owns_data);
  }

  return SparseBatch(std::move(batch_vectors), starting_id);
}

static DenseBatch wrapNumpyIntoDenseBatch(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& data,
    uint64_t starting_id) {
  const py::buffer_info data_buf = data.request();
  const auto shape = data_buf.shape;
  if (shape.size() != 2) {
    throw std::invalid_argument(
        "For now, Numpy dense data must be 2D (each row is a dense data "
        "vector).");
  }

  uint64_t num_vectors = static_cast<uint64_t>(shape.at(0));
  uint64_t dimension = static_cast<uint64_t>(shape.at(1));
  float* raw_data = static_cast<float*>(data_buf.ptr);

  std::vector<dataset::DenseVector> batch_vectors;
  for (uint64_t vec_id = 0; vec_id < num_vectors; vec_id++) {
    // owns_data = false because we don't want the numpy array to be deleted
    // if this batch (and thus the underlying vectors) get deleted
    bool owns_data = false;
    batch_vectors.emplace_back(dimension, raw_data + dimension * vec_id,
                               owns_data);
  }

  return DenseBatch(std::move(batch_vectors), starting_id);
}

static InMemoryDataset<SparseBatch> loadSVMDataset(const std::string& filename,
                                                   uint32_t batch_size) {
  auto start = std::chrono::high_resolution_clock::now();
  InMemoryDataset<SparseBatch> data(filename, batch_size,
                                    thirdai::dataset::SvmSparseBatchFactory{});
  auto end = std::chrono::high_resolution_clock::now();

  std::cout
      << "Read " << data.len() << " vectors in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;

  return data;
}

class PyFlash final : public Flash64 {
 public:
  explicit PyFlash(const HashFunction& function) : Flash64(function) {}

  PyFlash(const HashFunction& function, uint32_t reservoir_size)
      : Flash64(function, reservoir_size) {}

  void addDenseBatch(
      const py::array_t<float, py::array::c_style | py::array::forcecast>& data,
      uint64_t starting_id) {
    Flash64::addBatch(wrapNumpyIntoDenseBatch(data, starting_id));
  }

  void addSparseBatch(
      const std::vector<py::array_t<float, py::array::c_style |
                                               py::array::forcecast>>& values,
      const std::vector<
          py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>&
          indices,
      uint64_t starting_id) {
    Flash64::addBatch(wrapNumpyIntoSparseData(values, indices, starting_id));
  }

  py::array queryDenseBatch(
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          queries,
      uint32_t top_k) {
    bool pad_zeros = true;
    auto query_batch = wrapNumpyIntoDenseBatch(queries, 0);
    auto result = Flash64::queryBatch(query_batch, top_k, pad_zeros);
    return py::cast(result);
  }

  py::array querySparseBatch(
      const std::vector<py::array_t<
          float, py::array::c_style | py::array::forcecast>>& query_values,
      const std::vector<
          py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>&
          query_indices,
      uint32_t top_k) {
    bool pad_zeros = true;
    auto query_batch = wrapNumpyIntoSparseData(query_values, query_indices, 0);
    auto result = Flash64::queryBatch(query_batch, top_k, pad_zeros);
    return py::cast(result);
  }
};

}  // namespace thirdai::python

using thirdai::python::PyFlash;

// TODO(all): Figure out naming convention for python exposed classes and
// methods
// TODO(any): Add docstrings to methods
// TODO(any): Can we remove redudancy in the bindings?
PYBIND11_MODULE(thirdai, m) {  // NOLINT

  auto hashing_submodule = m.def_submodule("hashing");

  // TODO(josh): Add proper sparse data type

  py::class_<HashFunction>(
      hashing_submodule, "HashFunction",
      "Represents an abstract hash function that maps input DenseVectors and "
      "SparseVectors to sets of positive integers")
      .def("get_num_tables", &HashFunction::numTables,
           "Returns the number of hash tables in this hash function, which is "
           "equivalently the number of hashes that get returned by the "
           "function for each input.")
      .def("get_range", &HashFunction::range,
           "All hashes returned from this function will be >= 0 and <= "
           "get_range().");
  // TODO(josh): Add bindings for hashing numpy array and sparse data

  py::class_<DensifiedMinHash, HashFunction>(
      hashing_submodule, "MinHash",
      "A concrete implementation of a HashFunction that performs an extremly "
      "efficient minhash. A statistical estimator of jaccard similarity.")
      .def(py::init<uint32_t, uint32_t, uint32_t>(),
           py::arg("hashes_per_table"), py::arg("num_tables"),
           py::arg("range"));

  py::class_<FastSRP, HashFunction>(
      hashing_submodule, "SignedRandomProjection",
      "A concrete implementation of a HashFunction that performs an extremly "
      "efficient signed random projection. A statistical estimator of cossine "
      "similarity.")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
           py::arg("input_dim"), py::arg("hashes_per_table"),
           py::arg("num_tables"), py::arg("range") = UINT32_MAX);

#ifndef __clang__
  m.def("set_global_num_threads", &omp_set_num_threads,
        py::arg("max_num_threads"));
#endif

  auto dataset_submodule = m.def_submodule("dataset");

  py::class_<InMemoryDataset<SparseBatch>> _imsd_(dataset_submodule,
                                                  "InMemorySparseDataset");
  (void)_imsd_;  // To get rid of clang tidy error

  dataset_submodule.def(
      "loadClickThroughDataset", &thirdai::python::loadClickThorughDataset,
      py::arg("filename"), py::arg("batch_size"), py::arg("num_dense_features"),
      py::arg("num_categorical_features"));

  py::class_<
      thirdai::dataset::InMemoryDataset<thirdai::dataset::ClickThroughBatch>>
      _imctd_(dataset_submodule, "ClickThroughDataset");
  (void)_imctd_;  // To get rid of clang tidy error.

  dataset_submodule.def("loadSVMDataset", &thirdai::python::loadSVMDataset,
                        py::arg("filename"), py::arg("batch_size"));

  auto search_submodule = m.def_submodule("search");
  py::class_<PyFlash>(
      search_submodule, "MagSearch",
      "MagSearch is an index for performing near neighbor search. To use it, "
      "construct an index by passing in a hash function and then calling "
      "add() at least once to populate the index.")
      .def(py::init<HashFunction&, uint32_t>(), py::arg("hash_function"),
           py::arg("reservoir_size"),
           "Builds a MagSearch index where all hash "
           "buckets have a max size reservoir_size.")
      .def(py::init<HashFunction&>(), py::arg("hash_function"),
           "Builds a MagSearch index where buckets do not have a max size.")
      .def("add", &PyFlash::addDenseBatch, py::arg("dense_data"),
           py::arg("starting_index"),
           "Adds a dense numpy batch to the "
           "index, where each row represents a vector with sequential ids "
           "starting from the passed in starting_index.")
      .def("add", &PyFlash::addSparseBatch, py::arg("sparse_values"),
           py::arg("sparse_indices"), py::arg("starting_index"),
           "Adds a sparse batch batch to the "
           "index, where each corresponding pair of items from sparse_values "
           "and sparse_indices represents a sparse vector. The vectors have "
           "sequential ids starting from the passed in starting_index.")
      .def("query", &PyFlash::queryDenseBatch, py::arg("dense_queries"),
           py::arg("top_k") = 10,
           "Performs a batch query that returns the "
           "approximate top_k neighbors as a row for each of the passed in "
           "queries.")
      .def("query", &PyFlash::querySparseBatch, py::arg("sparse_query_values"),
           py::arg("sparse_query_indices"), py::arg("top_k") = 10,
           "Performs a batch query that returns the "
           "approximate top_k neighbors as a row for each of the passed in "
           "queries.");

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

  py::class_<thirdai::bolt::EmbeddingLayerConfig>(bolt_submodule,
                                                  "EmbeddingLayerConfig")
      .def(py::init<uint32_t, uint32_t, uint32_t>(),
           py::arg("num_embedding_lookups"), py::arg("lookup_size"),
           py::arg("log_embedding_block_size"));

  py::class_<thirdai::python::PyNetwork>(bolt_submodule, "Network")
      .def(py::init<std::vector<thirdai::bolt::FullyConnectedLayerConfig>,
                    uint64_t>(),
           py::arg("layers"), py::arg("input_dim"))
      .def("Train", &thirdai::python::PyNetwork::train, py::arg("train_data"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("rehash") = 0,
           py::arg("rebuild") = 0)
      .def("Test", &thirdai::python::PyNetwork::test, py::arg("test_data"),
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max())
      .def("UseSparseInference",
           &thirdai::python::PyNetwork::useSparseInference)
      .def("GetWeightMatrix", &thirdai::python::PyNetwork::getWeightMatrix,
           py::arg("layer_index"))
      .def("GetBiasVector", &thirdai::python::PyNetwork::getBiasVector,
           py::arg("layer_index"))
      .def("GetNumLayers", &thirdai::python::PyNetwork::getNumLayers)
      .def("GetLayerSizes", &thirdai::python::PyNetwork::getLayerSizes)
      .def("GetInputDim", &thirdai::python::PyNetwork::getInputDim)
      .def("GetActivationFunctions",
           &thirdai::python::PyNetwork::getActivationFunctions);

  py::class_<thirdai::python::PyDLRM>(bolt_submodule, "DLRM")
      .def(py::init<thirdai::bolt::EmbeddingLayerConfig,
                    std::vector<thirdai::bolt::FullyConnectedLayerConfig>,
                    std::vector<thirdai::bolt::FullyConnectedLayerConfig>,
                    uint32_t>(),
           py::arg("embedding_layer"), py::arg("bottom_mlp"),
           py::arg("top_mlp"), py::arg("input_dim"))
      .def("Train", &thirdai::python::PyDLRM::train, py::arg("train_data"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("rehash"),
           py::arg("rebuild"))
      .def("Test", &thirdai::python::PyDLRM::test, py::arg("test_data"));
}