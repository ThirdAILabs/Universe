
#include "LicenseWrapper.h"
#if THIRDAI_CHECK_LICENSE
#include <licensing/src/LicenseWithSignature.h>
#endif

namespace thirdai::licensing {

// Initialize this here to prevent linker errors, see
// https://stackoverflow.com/questions/185844/how-to-initialize-private-static-members-in-c
std::optional<std::string> LicenseWrapper::_license_path = {};

void LicenseWrapper::checkLicense() {
#if THIRDAI_CHECK_LICENSE
#pragma message( \
    "THIRDAI_CHECK_LICENSE is defined, adding license checking code")  // NOLINT
  LicenseWithSignature::findVerifyAndCheckLicense(_license_path);
#endif
}

void LicenseWrapper::setLicensePath(const std::string& license_path) {
  _license_path = license_path;
}

}  // namespace thirdai::licensing