#pragma once
#include <licensing/src/entitlements/Entitlements.h>
#include <licensing/src/methods/LicenseMethod.h>
#include <licensing/src/methods/file/SignedLicense.h>

namespace thirdai::licensing::file {

class FileMethod final : public LicenseMethod {
 public:
  FileMethod(std::string license_path, bool verbose);

  void checkLicense() override;

  LicenseState getLicenseState() override;

 private:
  std::string _license_path;
};

}  // namespace thirdai::licensing::file