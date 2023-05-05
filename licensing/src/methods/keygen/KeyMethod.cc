#include "KeyMethod.h"

namespace thirdai::licensing::keygen {

KeyMethod::KeyMethod(std::string api_key)
    : LicenseMethod(keygen::verifyKeyAndGetEntitlements(api_key),
                    licensing::LicenseMethodType::KEY),
      _api_key(std::move(api_key)){};

void KeyMethod::checkLicense() {
  Entitlements entitlements = keygen::verifyKeyAndGetEntitlements(_api_key);
  _entitlements = entitlements;
}

LicenseState KeyMethod::getLicenseState() {
  LicenseState license_state;
  license_state.key_state = _api_key;
  return license_state;
}

}  // namespace thirdai::licensing::keygen