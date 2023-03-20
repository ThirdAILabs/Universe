
#include "CheckLicense.h"
#include <dataset/src/DataSource.h>
#include <dataset/src/cold_start/ColdStartDataSource.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/Utils.h>
#include <licensing/src/file/SignedLicense.h>
#include <licensing/src/heartbeat/Heartbeat.h>
#include <licensing/src/keygen/KeygenCommunication.h>
#include <memory>
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
    const dataset::DataSourcePtr& training_source) {
  if (_entitlements.count(FULL_ACCESS_ENTITLEMENT)) {
    return;
  }

  // If the user just has a demo license and we are going to read in the dataset
  // from the resourceName, we require FileDataSources or ColdStartDataSources.
  // This prevents a user from extending the DataSource class in python and
  // making resourceName() point to a valid file, while the actual nextLine call
  // returns lines from some other file they want to train on.
  if (!dynamic_cast<dataset::FileDataSource*>(training_source.get()) &&
      !dynamic_cast<dataset::cold_start::ColdStartDataSource*>(
          training_source.get())) {
    throw exceptions::LicenseCheckException(
        "Can only train on file data sources with a demo license");
  }

  std::string file_path = training_source->resourceName();

  if (!_entitlements.count(sha256File(file_path))) {
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

}  // namespace thirdai::licensing