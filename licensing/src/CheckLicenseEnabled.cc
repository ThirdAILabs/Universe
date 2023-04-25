
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

// TODO(Kartik): Decide if we want to refactor these into a single
// std::unique_ptr<LicensingMethod> with a getEntitlements call. We would then
// set this in each call that starts licensing, and check the license there
// immediately upon the time it is set.
std::unique_ptr<HeartbeatThread> _heartbeat_thread = nullptr;
LicenseState _license_state;

std::optional<Entitlements> _entitlements;

void checkLicense() {
#pragma message( \
    "THIRDAI_CHECK_LICENSE is defined, adding license checking code")  // NOLINT

  if (_license_state.api_key_state.has_value()) {
    _entitlements =
        keygen::entitlementsFromKeygen(*_license_state.api_key_state);
    return;
  }

  if (_heartbeat_thread != nullptr) {
    _heartbeat_thread->verify();
    // For now heartbeat licenses are always full access because adding
    // entitlement support to our on prem license server is complicated.
    _entitlements = Entitlements({FULL_ACCESS_ENTITLEMENT});
    return;
  }

  if (_license_state.license_path_state.has_value()) {
    _entitlements = SignedLicense::entitlementsFromLicenseFile(
        _license_state.license_path_state.value(), /* verbose = */ false);
    return;
  }

  throw exceptions::LicenseCheckException(
      "Please first call either licensing.set_path, "
      "licensing.start_heartbeat, or licensing.activate.");
}

Entitlements entitlements() {
  if (!_entitlements.has_value()) {
    throw std::runtime_error(
        "Cannot get entitlements if we have not yet found a license.");
  }
  return _entitlements.value();
}

void activate(const std::string& api_key) {
  _license_state.api_key_state = api_key;
}

void deactivate() {
  _license_state.api_key_state = std::nullopt;
  _entitlements = std::nullopt;
}

void startHeartbeat(const std::string& heartbeat_url,
                    const std::optional<uint32_t>& heartbeat_timeout) {
  _heartbeat_thread =
      std::make_unique<HeartbeatThread>(heartbeat_url, heartbeat_timeout);
  _license_state.heartbeat_state = {heartbeat_url, heartbeat_timeout};
}

void endHeartbeat() {
  _heartbeat_thread = nullptr;
  _license_state.heartbeat_state = std::nullopt;
}

void setLicensePath(const std::string& license_path, bool verbose) {
  _license_state.license_path_state = license_path;
  // This verifies the license file so failures are visible here, and also so
  // that we print the value of the license. This also is the pattern we want
  // to move licensing to: see the above TODO for more details.
  SignedLicense::entitlementsFromLicenseFile(
      _license_state.license_path_state.value(), verbose);
}

LicenseState getLicenseState() { return _license_state; }

}  // namespace thirdai::licensing