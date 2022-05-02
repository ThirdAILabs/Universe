
#include "LicenseWrapper.h"
#if THIRDAI_CHECK_LICENSE
#include <licensing/src/LicenseWithSignature.h>
#endif

namespace thirdai::licensing {

void LicenseWrapper::checkLicense() {
#if THIRDAI_CHECK_LICENSE
  LicenseWithSignature::findVerifyAndCheckLicense();
#endif
}

}  // namespace thirdai::licensing