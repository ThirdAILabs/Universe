
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
#if THIRDAI_CHECK_LICENSE

static std::optional<std::string> _license_path = {};
static std::optional<std::string> _api_key = {};
static std::unordered_set<std::string> _entitlements = {};

static std::unique_ptr<HeartbeatThread> _heartbeat_thread = nullptr;

void checkLicense() {
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
}

void verifyAllowedDataset(const dataset::DataLoaderPtr& data_loader) {
  if (_entitlements.count(FULL_ACCESS_ENTITLEMENT)) {
    return;
  }

  std::string dataset_hash =
      sha256File(/* filename = */ data_loader->resourceName());
  if (!_entitlements.count(dataset_hash)) {
    throw std::runtime_error(
        "This dataset is not authorized under this license.");
  }
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

#else

void checkLicense() {}

void verifyAllowedDataset(const dataset::DataLoaderPtr& data_loader) {
  (void)data_loader;
}

void activate(const std::string& api_key) { (void)api_key; }

void deactivate() {}

void startHeartbeat(const std::string& heartbeat_url,
                    const std::optional<uint32_t>& heartbeat_timeout) {
  (void)heartbeat_url;
  (void)heartbeat_timeout;
}

void endHeartbeat() {}

void setLicensePath(const std::string& license_path) { (void)license_path; }

#endif
}  // namespace thirdai::licensing