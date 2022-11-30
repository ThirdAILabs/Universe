
#include "LicenseWrapper.h"
#include <optional>
#if THIRDAI_CHECK_LICENSE
#include <licensing/src/CheckLicense.h>
#endif

namespace thirdai::licensing {

// Initialize these here to prevent linker errors, see
// https://stackoverflow.com/questions/185844/how-to-initialize-private-static-members-in-c
std::optional<std::string> LicenseWrapper::_license_path = {};
std::optional<std::string> LicenseWrapper::_api_key = {};

void LicenseWrapper::checkLicenseWrapper() {
#if THIRDAI_CHECK_LICENSE
#pragma message( \
    "THIRDAI_CHECK_LICENSE is defined, adding license checking code")  // NOLINT
  checkLicense(_api_key, _license_path);
#endif
}

void LicenseWrapper::activate(const std::string& api_key) {
  _api_key = api_key;
}

void LicenseWrapper::setLicensePath(const std::string& license_path) {
  _license_path = license_path;
}

void LicenseWrapper::deactivate() { _license_path = std::nullopt; }

}  // namespace thirdai::licensing