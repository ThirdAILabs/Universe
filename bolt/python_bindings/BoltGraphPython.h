#pragma once

#include <bolt/src/graph/Graph.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

void createBoltGraphSubmodule(py::module_& bolt_submodule);

py::tuple dagPredictPythonWrapper(BoltGraph& model,
                                  const dataset::BoltDatasetList& data,
                                  const dataset::BoltTokenDatasetList& tokens,
                                  const dataset::BoltDatasetPtr& labels,
                                  const PredictConfig& predict_config);

}  // namespace thirdai::bolt::python
