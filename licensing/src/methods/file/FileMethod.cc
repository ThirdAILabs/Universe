#include "FileMethod.h"

namespace thirdai::licensing::file {

FileMethod::FileMethod(std::string license_path, bool verbose)
    : LicenseMethod(
          SignedLicense::verifyPathAndGetEntitlements(license_path, verbose),
          licensing::LicenseMethodType::FILE),
      _license_path(std::move(license_path)) {}

void FileMethod::checkLicense() {
  Entitlements entitlements = SignedLicense::verifyPathAndGetEntitlements(
      _license_path, /* verbose = */ false);
  _entitlements = entitlements;
}

LicenseState FileMethod::getLicenseState() {
  LicenseState license_state;
  license_state.file_state = _license_path;
  return license_state;
}

}  // namespace thirdai::licensing::file