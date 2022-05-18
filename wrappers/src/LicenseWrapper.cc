
#include "LicenseWrapper.h"
#if THIRDAI_CHECK_LICENSE
#include <licensing/src/LicenseWithSignature.h>
#endif

namespace thirdai::licensing {

void LicenseWrapper::checkLicense(
    const std::optional<std::string>& license_path) {
  (void)license_path;
#if THIRDAI_CHECK_LICENSE
#pragma message( \
    "THIRDAI_CHECK_LICENSE is defined, adding license checking code")  // NOLINT
  LicenseWithSignature::findVerifyAndCheckLicense(license_path);
#endif
}

}  // namespace thirdai::licensing