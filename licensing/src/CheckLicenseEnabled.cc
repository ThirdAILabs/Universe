
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

FinegrainedAccessToken::FinegrainedAccessToken(
    const std::string& train_file_path) {
  if (_entitlements.count(FULL_ACCESS_ENTITLEMENT)) {
    return;
  }

  if (!_entitlements.count(sha256File(train_file_path))) {
    throw exceptions::LicenseCheckException(
        "This dataset is not authorized under this license.");
  }
}

FinegrainedAccessToken::FinegrainedAccessToken() {
  if (!_entitlements.count(FULL_ACCESS_ENTITLEMENT)) {
    throw std::runtime_error(
        "You must have a full license to perform this operation.");
  }
}

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

void verifyCanSaveAndLoad() {
  if (!_entitlements.count(FULL_ACCESS_ENTITLEMENT)) {
    throw exceptions::LicenseCheckException(
        "You must have a full license to save and load models.");
  }
}

}  // namespace thirdai::licensing