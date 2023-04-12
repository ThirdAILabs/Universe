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

  py::class_<LicenseState>(licensing_submodule, "LicenseState")
      // https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support
      .def(py::pickle(
          [](const LicenseState& s) {  // __getstate__
            return py::make_tuple(s.api_key_state, s.heartbeat_state,
                                  s.license_path_state);
          },
          [](const py::tuple& t) {  // __setstate__
            if (t.size() != 3) {
              throw std::runtime_error("Invalid state!");
            }

            LicenseState s;
            s.api_key_state = t[0].cast<std::optional<std::string>>();
            s.heartbeat_state = t[1].cast<std::optional<
                std::pair<std::string, std::optional<uint32_t>>>>();
            s.license_path_state = t[2].cast<std::optional<std::string>>();

            return s;

          }));

  licensing_submodule.def("end_heartbeat", &thirdai::licensing::endHeartbeat,
                          "Ends the current ThirdAI heartbeat.");

  licensing_submodule.def(
      "_get_license_state", &thirdai::licensing::getLicenseState,
      "Gets a summary of all current ThirdAI licensing metadata.");

  licensing_submodule.def("_set_license_state",
                          &thirdai::licensing::setLicenseState,
                          py::arg("license_state"),
                          "Sets a summary of all current ThirdAI licensing "
                          "metadata, as returned by _get_license_info.");
}

}  // namespace thirdai::licensing::python
