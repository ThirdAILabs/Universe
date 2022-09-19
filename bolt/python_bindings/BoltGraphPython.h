#pragma once

#include "ConversionUtils.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <compression/python_bindings/ConversionUtils.h>
#include <compression/src/CompressedVector.h>
#include <compression/src/CompressionFactory.h>
#include <dataset/src/Datasets.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <memory>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::bolt::python {

void createBoltGraphSubmodule(py::module_& bolt_submodule);

void createCallbacksSubmodule(py::module_& graph_submodule);

py::tuple dagPredictPythonWrapper(BoltGraph& model,
                                  const dataset::BoltDatasetList& data,
                                  const dataset::BoltDatasetPtr& labels,
                                  const PredictConfig& predict_config);

py::tuple dagGetInputGradientSingleWrapper(
    const std::pair<std::optional<std::vector<uint32_t>>, std::vector<float>>&
        gradients);
using ParameterArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

using SerializedCompressedVector =
    py::array_t<char, py::array::c_style | py::array::forcecast>;

class ParameterReference {
  using CompressedVector = thirdai::compression::CompressedVector<float>;

 public:
  ParameterReference(float* params, std::vector<uint32_t> dimensions)
      : _params(params), _dimensions(std::move(dimensions)) {
    _total_dim = dimensionProduct(_dimensions);
  }

  ParameterArray copy() const {
    float* params_copy = new float[_total_dim];
    std::copy(_params, _params + _total_dim, params_copy);

    py::capsule free_when_done(
        params_copy, [](void* ptr) { delete static_cast<float*>(ptr); });

    return ParameterArray(_dimensions, params_copy, free_when_done);
  }

  ParameterArray get() const { return ParameterArray(_dimensions, _params); }

  void set(const py::object& new_params) {
    if (py::isinstance<ParameterArray>(new_params)) {
      ParameterArray new_array = py::cast<ParameterArray>(new_params);
      checkNumpyArrayDimensions(_dimensions, new_params);
      std::copy(new_array.data(), new_array.data() + _total_dim, _params);
    } else if (py::isinstance<SerializedCompressedVector>(new_params)) {
      std::unique_ptr<CompressedVector> compressed_vector =
          thirdai::compression::python::deserializeCompressedVector(
              py::cast<SerializedCompressedVector>(new_params).data());

      std::vector<float> full_gradients = compressed_vector->decompress();
      std::copy(full_gradients.data(), full_gradients.data() + _total_dim,
                _params);
    } else {
      throw std::invalid_argument(
          "Cannot set parameters from an unsupported Python datatype");
    }
  }

  SerializedCompressedVector compress(const std::string& compression_scheme,
                                      float compression_density,
                                      uint32_t seed_for_hashing,
                                      uint32_t sample_population_size) {
    std::unique_ptr<CompressedVector> compressed_vector =
        thirdai::compression::compress(
            _params, static_cast<uint32_t>(_total_dim), compression_scheme,
            compression_density, seed_for_hashing, sample_population_size);

    char* serialized_compressed_vector =
        new char[compressed_vector->serialized_size()];

    thirdai::compression::python::serializeCompressedVector(
        compressed_vector, serialized_compressed_vector);
    py::capsule free_when_done(serialized_compressed_vector, [](void* ptr) {
      delete static_cast<char*>(ptr);
    });
    return SerializedCompressedVector(compressed_vector->serialized_size(),
                                      serialized_compressed_vector,
                                      free_when_done);
  }

  static SerializedCompressedVector concat(
      const py::object& compressed_vectors) {
    if (py::isinstance<py::list>(compressed_vectors)) {
      std::unique_ptr<CompressedVector> concatenated_compressed_vector =
          thirdai::compression::concat(
              thirdai::compression::python::convertPyListToCompressedVectors(
                  compressed_vectors));

      char* serialized_compressed_vector =
          new char[concatenated_compressed_vector->serialized_size()];
      thirdai::compression::python::serializeCompressedVector(
          concatenated_compressed_vector, serialized_compressed_vector);
      py::capsule free_when_done(serialized_compressed_vector, [](void* ptr) {
        delete static_cast<char*>(ptr);
      });
      return SerializedCompressedVector(
          concatenated_compressed_vector->serialized_size(),
          serialized_compressed_vector, free_when_done);
    }
    throw std::invalid_argument(
        "Cannot concat compressed vectors from unsupported data type, expects "
        "list of compressed vectors.");
  }

 private:
  static uint64_t dimensionProduct(const std::vector<uint32_t>& dimensions) {
    uint64_t product = 1;
    for (uint32_t dim : dimensions) {
      product *= dim;
    }
    return product;
  }

  float* _params;
  std::vector<uint32_t> _dimensions;
  uint64_t _total_dim;
};

}  // namespace thirdai::bolt::python
