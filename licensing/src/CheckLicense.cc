
#include "CheckLicense.h"
#include <dataset/src/DataLoader.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#if THIRDAI_CHECK_LICENSE
#include <licensing/src/file/SignedLicense.h>
#include <licensing/src/heartbeat/Heartbeat.h>
#include <licensing/src/keygen/KeygenCommunication.h>
#include <licensing/src/utils.h>
#endif

namespace thirdai::licensing {

static const std::string FULL_ACCESS_ENTITLEMENT = "FULL_ACCESS";

static std::optional<std::string> _license_path = {};
static std::optional<std::string> _api_key = {};
static std::unordered_set<std::string> _entitlements = {};

#ifdef THIRDAI_CHECK_LICENSE
static std::unique_ptr<HeartbeatThread> _heartbeat_thread = nullptr;
#endif

void checkLicense() {
#if THIRDAI_CHECK_LICENSE
#pragma message( \
    "THIRDAI_CHECK_LICENSE is defined, adding license checking code")  // NOLINT

  if (_api_key.has_value()) {
    _entitlements = verifyWithKeygen(*_api_key);
    return;
  }

  if (_heartbeat_thread != nullptr) {
    _heartbeat_thread->verify();
  }

  SignedLicense::findVerifyAndCheckLicense(_license_path);
  _entitlements.insert(FULL_ACCESS_ENTITLEMENT);

#endif
}

void verifyAllowedDataset(const dataset::DataLoaderPtr& data_loader) {
#ifndef THIRDAI_CHECK_LICENSE
  (void)data_loader;
  return;
#else
  if (_entitlements.count(FULL_ACCESS_ENTITLEMENT)) {
    return;
  }

  std::string dataset_hash =
      sha256File(/* filename = */ data_loader->resourceName());
  if (!_entitlements.count(dataset_hash)) {
    throw std::runtime_error(
        "This dataset is not authorized under this license.");
  }
#endif
}

void activate(const std::string& api_key) { _api_key = api_key; }

void deactivate() {
  _api_key = std::nullopt;
  _entitlements.clear();
}

void startHeartbeat(const std::string& heartbeat_url,
                    const std::optional<uint32_t>& heartbeat_timeout) {
  (void)heartbeat_url;
  (void)heartbeat_timeout;
#if THIRDAI_CHECK_LICENSE
  _heartbeat_thread =
      std::make_unique<HeartbeatThread>(heartbeat_url, heartbeat_timeout);
#endif
}

// TODO(Josh): clean up all of these ifdefs into a single ifdef somewhere

void endHeartbeat() {
#if THIRDAI_CHECK_LICENSE
  _heartbeat_thread = nullptr;
#endif
}

void setLicensePath(const std::string& license_path) {
  _license_path = license_path;
}
}  // namespace thirdai::licensing