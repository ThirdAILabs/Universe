#include "LicensingPython.h"
#include <licensing/src/CheckLicense.h>
#include <pybind11/stl.h>

namespace thirdai::licensing::python {

void createLicensingSubmodule(py::module_& module) {
  // Everything in this submodule is exposed to users.
  auto licensing_submodule = module.def_submodule("licensing");

  licensing_submodule.def(
      "set_path", &thirdai::licensing::setLicensePath, py::arg("license_path"),
      py::arg("verbose") = false,
      "Set a license filepath for any future calls to ThirdAI functions.");

  licensing_submodule.def(
      "activate", &thirdai::licensing::activate, py::arg("api_key"),
      "Set a ThirdAI API access key to authenticate future calls to ThirdAI "
      "functions.");

  licensing_submodule.def(
      "start_heartbeat", &thirdai::licensing::startHeartbeat,
      py::arg("license_server_url"),
      py::arg("heartbeat_timeout") = std::nullopt,
      "Starts a ThirdAI heartbeat endpoint to remain authenticated for future "
      "calls to ThirdAI functions.");

  licensing_submodule.def(
      "deactivate", &thirdai::licensing::deactivate,
      "Deactivate the currently active license. Future calls to "
      "ThirdAI functions may fail.");

  py::class_<LicenseState>(licensing_submodule, "LicenseState")
      // https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support
      .def(py::pickle(
          [](const LicenseState& s) {  // __getstate__
            return py::make_tuple(s.key_state, s.local_server_state,
                                  s.file_state);
          },
          [](const py::tuple& t) {  // __setstate__
            if (t.size() != 3) {
              throw std::runtime_error("Invalid state!");
            }

            LicenseState s;
            s.key_state = t[0].cast<std::optional<std::string>>();
            s.local_server_state = t[1].cast<std::optional<
                std::pair<std::string, std::optional<uint32_t>>>>();
            s.file_state = t[2].cast<std::optional<std::string>>();

            return s;

          }));

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
