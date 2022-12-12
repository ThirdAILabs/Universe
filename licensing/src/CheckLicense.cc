
#include "CheckLicense.h"
#include <dataset/src/DataLoader.h>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#if THIRDAI_CHECK_LICENSE
#include <licensing/src/file/SignedLicense.h>
#include <licensing/src/keygen/KeygenCommunication.h>
#include <licensing/src/utils.h>
#endif

namespace thirdai::licensing {

static const std::string FULL_ACCESS_ENTITLEMENT = "FULL_ACCESS";

static std::optional<std::string> _license_path = {};
static std::optional<std::string> _api_key = {};
static std::unordered_set<std::string> _entitlements = {};

void checkLicense() {
#if THIRDAI_CHECK_LICENSE
#pragma message( \
    "THIRDAI_CHECK_LICENSE is defined, adding license checking code")  // NOLINT

  if (_api_key.has_value()) {
    _entitlements = verifyWithKeygen(*_api_key);
    return;
  }

  SignedLicense::findVerifyAndCheckLicense(_license_path);
  _entitlements.insert(FULL_ACCESS_ENTITLEMENT);

#endif
}

void activate(const std::string& api_key) { _api_key = api_key; }

void setLicensePath(const std::string& license_path) {
  _license_path = license_path;
}

void deactivate() {
  _api_key = std::nullopt;
  _entitlements.clear();
}

void verifyAllowedDataset(const dataset::DataLoaderPtr& data_loader) {
#ifndef THIRDAI_CHECK_LICENSE
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

}  // namespace thirdai::licensing