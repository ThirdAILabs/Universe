#pragma once

#include "ConversionUtils.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

void createBoltGraphSubmodule(py::module_& bolt_submodule);

py::tuple dagPredictPythonWrapper(BoltGraph& model,
                                  const dataset::BoltDatasetList& data,
                                  const dataset::BoltTokenDatasetList& tokens,
                                  const dataset::BoltDatasetPtr& labels,
                                  const PredictConfig& predict_config);

using ParameterArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

class ParameterReference {
 public:
  ParameterReference(float* params, std::vector<uint32_t> dimensions)
      : _params(params), _dimensions(std::move(dimensions)), _total_dim(1) {
    for (uint64_t dim : _dimensions) {
      _total_dim *= dim;
    }
  }

  ParameterArray copy() const {
    float* params_copy = new float[_total_dim];
    std::copy(_params, _params + _total_dim, params_copy);

    py::capsule free_when_done(
        params_copy, [](void* ptr) { delete static_cast<float*>(ptr); });

    return ParameterArray(_dimensions, getStrides(), params_copy,
                          free_when_done);
  }

  ParameterArray get() const {
    return ParameterArray(_dimensions, getStrides(), _params);
  }

  void set(const ParameterArray& new_params) {
    checkNumpyArrayDimensions(_dimensions, new_params);

    std::copy(new_params.data(), new_params.data() + _total_dim, _params);
  }

 private:
  std::vector<uint32_t> getStrides() const {
    std::vector<uint32_t> strides(_dimensions.size());

    strides.back() = sizeof(float);
    for (uint32_t i = _dimensions.size() - 1; i > 0; i--) {
      strides[i - 1] = _dimensions[i] * strides[i];
    }
    return strides;
  }

  float* _params;
  std::vector<uint32_t> _dimensions;
  uint64_t _total_dim;
};

}  // namespace thirdai::bolt::python
