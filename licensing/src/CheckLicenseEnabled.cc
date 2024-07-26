
#include "CheckLicense.h"
#include <dataset/src/DataSource.h>
#include <dataset/src/cold_start/ColdStartDataSource.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/Utils.h>
#include <licensing/src/entitlements/Entitlements.h>
#include <licensing/src/methods/LicenseMethod.h>
#include <licensing/src/methods/file/FileMethod.h>
#include <licensing/src/methods/file/SignedLicense.h>
// #include <licensing/src/methods/heartbeat/Heartbeat.h>
// #include <licensing/src/methods/heartbeat/LocalServerMethod.h>
// #include <licensing/src/methods/keygen/KeyMethod.h>
// #include <licensing/src/methods/keygen/KeygenCommunication.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

// This file is only linked when the feature flag THIRDAI_CHECK_LICENSE is True

namespace thirdai::licensing {

std::unique_ptr<LicenseMethod> _licensing_method = nullptr;

void checkLicense() {
  // if (_licensing_method == nullptr) {
  //   throw exceptions::LicenseCheckException(
  //       "Please first call either licensing.set_path, "
  //       "licensing.start_heartbeat, or licensing.activate with a valid "
  //       "license.");
  // }

  // _licensing_method->checkLicense();
}

Entitlements entitlements() {
  if (_licensing_method == nullptr) {
    throw exceptions::LicenseCheckException(
        "Cannot get entitlements if licensing is not initialized yet.");
  }
  return _licensing_method->getEntitlements();
}

void activate(std::string api_key) {
  // _licensing_method = std::make_unique<keygen::KeyMethod>(std::move(api_key));
  (void)api_key;
}

void startHeartbeat(std::string heartbeat_url,
                    std::optional<uint32_t> heartbeat_timeout) {
  // _licensing_method = std::make_unique<heartbeat::LocalServerMethod>(
  //     std::move(heartbeat_url), heartbeat_timeout);
  (void)heartbeat_timeout;
  (void)heartbeat_url;
}

void setLicensePath(std::string license_path, bool verbose) {
  _licensing_method =
      std::make_unique<file::FileMethod>(std::move(license_path), verbose);
}

void deactivate() { _licensing_method = nullptr; }

LicenseState getLicenseState() {
  if (_licensing_method == nullptr) {
    throw exceptions::LicenseCheckException(
        "Cannot get license state if licensing is not initialized yet.");
  }
  return _licensing_method->getLicenseState();
}

void setLicenseState(const LicenseState& state) {
  if (state.key_state) {
    activate(state.key_state.value());
  } else if (state.local_server_state) {
    auto local_server_state_value = state.local_server_state.value();
    startHeartbeat(local_server_state_value.first,
                   local_server_state_value.second);
  } else if (state.file_state) {
    setLicensePath(state.file_state.value());
  }
}

}  // namespace thirdai::licensing