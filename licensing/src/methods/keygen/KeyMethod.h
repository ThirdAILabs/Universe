#pragma once
#include <licensing/src/entitlements/Entitlements.h>
#include <licensing/src/methods/LicenseMethod.h>
#include <licensing/src/methods/keygen/KeygenCommunication.h>

namespace thirdai::licensing::keygen {

class KeyMethod final : public LicenseMethod {
 public:
  explicit KeyMethod(std::string api_key);

  void checkLicense() final;

  LicenseState getLicenseState() final;

 private:
  std::string _api_key;
};

}  // namespace thirdai::licensing::keygen