
#include "CheckLicense.h"
#include <optional>
#if THIRDAI_CHECK_LICENSE
#include <licensing/src/file/SignedLicense.h>
#include <licensing/src/keygen/KeygenCommunication.h>
#endif

namespace thirdai::licensing {

std::optional<std::string> _license_path = {};
std::optional<std::string> _api_key = {};

void CheckLicense::checkLicenseWrapper() {
#if THIRDAI_CHECK_LICENSE
#pragma message( \
    "THIRDAI_CHECK_LICENSE is defined, adding license checking code")  // NOLINT

  if (_api_key.has_value()) {
    verifyWithKeygen(*_api_key);
    return;
  }

  SignedLicense::findVerifyAndCheckLicense(_license_path);

#endif
}

void CheckLicense::activate(const std::string& api_key) { _api_key = api_key; }

void CheckLicense::setLicensePath(const std::string& license_path) {
  _license_path = license_path;
}

void CheckLicense::deactivate() { _api_key = std::nullopt; }

}  // namespace thirdai::licensing