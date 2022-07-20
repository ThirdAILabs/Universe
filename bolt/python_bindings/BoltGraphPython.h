#pragma once

#include <bolt/src/graph/Graph.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

void createBoltGraphSubmodule(py::module_& bolt_submodule);

MetricData dagTrainPythonWrapper(
    BoltGraph& model, const py::object& data,
    const std::shared_ptr<dataset::InMemoryDataset<dataset::BoltTokenBatch>>&
        token_data,
    const py::object& labels, const TrainConfig& train_config);

py::tuple dagPredictPythonWrapper(
    BoltGraph& model, const py::object& data,
    const std::shared_ptr<dataset::InMemoryDataset<dataset::BoltTokenBatch>>&
        token_data,
    const py::object& labels, const PredictConfig& predict_config);

}  // namespace thirdai::bolt::python
