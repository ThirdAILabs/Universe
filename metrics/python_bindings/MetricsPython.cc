#include "MetricsDocs.h"
#include <metrics/src/PrometheusClient.h>
#include <pybind11/stl.h>
#include <optional>
#include <string>

namespace thirdai::metrics::python {

namespace py = pybind11;

void createMetricsSubmodule(py::module_& thirdai_module) {
  py::module_ submodule = thirdai_module.def_submodule("metrics");

  submodule.def("start_metrics", &createGlobalMetricsClient, py::arg("port") = THIRDAI_DEFAULT_METRICS_PORT, docs::START_METRICS);

  submodule.def("stop_metrics", &stopGlobalMetricsClient, docs::STOP_METRICS);

}

}  // namespace thirdai::metrics::python