#pragma once

#include "ConversionUtils.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <dataset/src/Datasets.h>
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

py::tuple dagGetInputGradientsWrapper(
    BoltGraph& model, const dataset::BoltDatasetPtr& input_data,
    const dataset::BoltTokenDatasetPtr& input_tokens, bool best_index = true,
    const std::vector<uint32_t>& required_labels = std::vector<uint32_t>());
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

  void set(const ParameterArray& new_params) {
    checkNumpyArrayDimensions(_dimensions, new_params);

    std::copy(new_params.data(), new_params.data() + _total_dim, _params);
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
