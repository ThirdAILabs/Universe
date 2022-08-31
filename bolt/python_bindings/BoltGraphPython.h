#pragma once

#include "ConversionUtils.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <compression/python_bindings/ConversionUtils.h>
#include <compression/src/CompressedVector.h>
#include <dataset/src/Datasets.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

void createBoltGraphSubmodule(py::module_& bolt_submodule);

py::tuple dagPredictPythonWrapper(BoltGraph& model,
                                  const dataset::BoltDatasetList& data,
                                  const dataset::BoltDatasetPtr& labels,
                                  const PredictConfig& predict_config);

py::tuple dagGetInputGradientSingleWrapper(
    const std::pair<std::optional<std::vector<uint32_t>>, std::vector<float>>&
        gradients);
using ParameterArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

class ParameterReference {
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
    if (py::isinstance<py::array>(new_params)) {
      ParameterArray new_array = py::cast<ParameterArray>(new_params);
      checkNumpyArrayDimensions(_dimensions, new_params);
      std::copy(new_array.data(), new_array.data() + _total_dim, _params);
    } else {
      using CompressedVector = thirdai::compression::CompressedVector<float>;
      std::unique_ptr<CompressedVector> compressed_vector =
          thirdai::compression::python::convertPyDictToCompressedVector(
              new_params);
      std::vector<float> full_gradients = compressed_vector->decompress();
      std::copy(full_gradients.data(), full_gradients.data() + _total_dim,
                _params);
    }
  }

  py::dict compress(const std::string& compression_scheme,
                    float compression_density, int seed_for_hashing) {
    return thirdai::compression::python::convertCompressedVectorToPyDict(
        thirdai::compression::compress(
            _params, static_cast<uint32_t>(_total_dim), compression_scheme,
            compression_density, seed_for_hashing));
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
