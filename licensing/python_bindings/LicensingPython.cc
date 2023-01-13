#include "LicensingPython.h"
#include <licensing/src/CheckLicense.h>
#include <pybind11/stl.h>

namespace thirdai::licensing::python {

void createLicensingSubmodule(py::module_& module) {
  // Everything in this submodule is exposed to users.
  auto licensing_submodule = module.def_submodule("licensing");

  licensing_submodule.def(
      "set_path", &thirdai::licensing::setLicensePath, py::arg("license_path"),
      "Set a license filepath for any future calls to ThirdAI functions. "
      "License file verification will be treated as a fallback if activate "
      "has not been called.");

  licensing_submodule.def(
      "activate", &thirdai::licensing::activate, py::arg("api_key"),
      "Set a ThirdAI API access key to authenticate future calls to ThirdAI "
      "functions.");

  licensing_submodule.def(
      "deactivate", &thirdai::licensing::deactivate,
      "Remove the currently stored ThirdAI access key. Future calls to "
      "ThirdAI functions may fail.");

  licensing_submodule.def(
      "start_heartbeat", &thirdai::licensing::startHeartbeat,
      py::arg("license_server_url"),
      py::arg("heartbeat_timeout") = std::nullopt,
      "Starts a ThirdAI heartbeat endpoint to remain authenticated for future "
      "calls to ThirdAI functions.");

  licensing_submodule.def("end_heartbeat", &thirdai::licensing::endHeartbeat,
                          "Ends the current ThirdAI heartbeat.");
}

}  // namespace thirdai::licensing::python
