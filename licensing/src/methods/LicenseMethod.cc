#include "LicenseMethod.h"

namespace thirdai::licensing {

LicenseMethod::LicenseMethod(Entitlements entitlements,
                             LicenseMethodType license_method_type)
    : _entitlements(std::move(entitlements)),
      _license_method_type(license_method_type) {}

Entitlements LicenseMethod::getEntitlements() { return _entitlements; }

LicenseMethodType LicenseMethod::getLicenseMethodType() {
  return _license_method_type;
}

}  // namespace thirdai::licensing
