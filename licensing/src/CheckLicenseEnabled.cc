
#include "CheckLicense.h"
#include <dataset/src/DataSource.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/file/SignedLicense.h>
#include <licensing/src/heartbeat/Heartbeat.h>
#include <licensing/src/keygen/KeygenCommunication.h>
#include <licensing/src/utils.h>
#include <optional>
#include <stdexcept>
#include <unordered_set>

// This file is only linked when the feature flag THIRDAI_CHECK_LICENSE is True

namespace thirdai::licensing {

static const std::string FULL_ACCESS_ENTITLEMENT = "FULL_ACCESS";

static std::optional<std::string> _license_path = {};
static std::optional<std::string> _api_key = {};
static std::unordered_set<std::string> _entitlements = {};

static std::unique_ptr<HeartbeatThread> _heartbeat_thread = nullptr;

void assertUserHasFullAccess() {
  if (!_entitlements.count(FULL_ACCESS_ENTITLEMENT)) {
    throw exceptions::LicenseCheckException(
        "You must have a full license to perform this operation.");
  }
}

TrainPermissionsToken::TrainPermissionsToken(
    const std::string& train_file_path) {
  if (_entitlements.count(FULL_ACCESS_ENTITLEMENT)) {
    return;
  }

  // This handles the rare case where someone is "illegally" trying to train on
  // a data source like S3 with a demo license (otherwise without this check,
  // we might get a segfault or weird error from cryptopp).
  if (!std::filesystem::exists(train_file_path)) {
    throw exceptions::LicenseCheckException(
        "Could not find a local file corresponding to the passed in data "
        "source, so cannot validate the dataset with the demo license.");
  }

  if (!_entitlements.count(sha256File(train_file_path))) {
    throw exceptions::LicenseCheckException(
        "This dataset is not authorized under this license.");
  }
}

TrainPermissionsToken::TrainPermissionsToken() { assertUserHasFullAccess(); }

void checkLicense() {
#pragma message( \
    "THIRDAI_CHECK_LICENSE is defined, adding license checking code")  // NOLINT

  if (_api_key.has_value()) {
    _entitlements = verifyWithKeygen(*_api_key);
    return;
  }

  if (_heartbeat_thread != nullptr) {
    _heartbeat_thread->verify();
    return;
  }

  SignedLicense::findVerifyAndCheckLicense(_license_path);
  _entitlements.insert(FULL_ACCESS_ENTITLEMENT);
}

void activate(const std::string& api_key) { _api_key = api_key; }

void deactivate() {
  _api_key = std::nullopt;
  _entitlements.clear();
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

void disableForDemoLicenses() { assertUserHasFullAccess(); }

}  // namespace thirdai::licensing