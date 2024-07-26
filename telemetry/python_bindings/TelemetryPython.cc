#include <pybind11/stl.h>
#include <telemetry/src/PrometheusClient.h>
#include <utils/UUID.h>
#include <optional>
#include <string>

namespace thirdai::telemetry::python {

namespace py = pybind11;

void createTelemetrySubmodule(py::module_& thirdai_module) {
  py::module_ submodule = thirdai_module.def_submodule("telemetry");

  submodule.def("start", &createGlobalTelemetryClient,
                py::arg("port") = THIRDAI_DEFAULT_TELEMETRY_PORT);

  submodule.def("stop", &stopGlobalTelemetryClient);

  submodule.def("uuid", []() { return utils::uuid::THIRDAI_UUID; });
}

}  // namespace thirdai::telemetry::python