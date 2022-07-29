#pragma once

#include <bolt/src/graph/Graph.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <dataset/src/Datasets.h>
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

}  // namespace thirdai::bolt::python
