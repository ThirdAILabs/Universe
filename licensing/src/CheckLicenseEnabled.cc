
#include "CheckLicense.h"
#include <dataset/src/DataSource.h>
#include <dataset/src/cold_start/ColdStartDataSource.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/Utils.h>
#include <licensing/src/entitlements/Entitlements.h>
#include <licensing/src/methods/file/SignedLicense.h>
#include <licensing/src/methods/heartbeat/Heartbeat.h>
#include <licensing/src/methods/keygen/KeygenCommunication.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

// This file is only linked when the feature flag THIRDAI_CHECK_LICENSE is True

namespace thirdai::licensing {

std::optional<std::string> _license_path = std::nullopt;
std::optional<std::string> _api_key = std::nullopt;
std::optional<Entitlements> _entitlements;

std::unique_ptr<HeartbeatThread> _heartbeat_thread = nullptr;

void checkLicense() {
#pragma message( \
    "THIRDAI_CHECK_LICENSE is defined, adding license checking code")  // NOLINT

  if (_api_key.has_value()) {
    _entitlements = Entitlements(verifyWithKeygen(*_api_key));
    return;
  }

  if (_heartbeat_thread != nullptr) {
    _heartbeat_thread->verify();
    _entitlements = Entitlements({FULL_ACCESS_ENTITLEMENT});
    return;
  }

  if (_license_path.has_value()) {
    _entitlements =
        Entitlements({SignedLicense::verifyLicenseFile(_license_path.value())});
    return;
  }

  throw exceptions::LicenseCheckException(
      "Please first call either licensing.set_license_path, "
      "licensing.start_heartbeat, or licensing.activate.");
}

Entitlements entitlements() {
  if (!_entitlements.has_value()) {
    throw std::runtime_error(
        "Cannot get entitlements if we have not yet found a license.");
  }
  return _entitlements.value();
}

void activate(const std::string& api_key) { _api_key = api_key; }

void deactivate() {
  _api_key = std::nullopt;
  _entitlements = std::nullopt;
}

void startHeartbeat(const std::string& heartbeat_url,
                    const std::optional<uint32_t>& heartbeat_timeout) {
  _heartbeat_thread =
      std::make_unique<HeartbeatThread>(heartbeat_url, heartbeat_timeout);
}

void endHeartbeat() { _heartbeat_thread = nullptr; }

void setLicensePath(const std::string& license_path) {
  _license_path = license_path;
}

}  // namespace thirdai::licensing