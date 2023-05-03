#pragma once
#include <licensing/src/entitlements/Entitlements.h>
#include <string>

namespace thirdai::licensing {

enum class LicenseMethodType { KEY, FILE, SERVER };

struct LicenseState {
  std::optional<std::string> key_state;
  std::optional<std::pair<std::string, std::optional<uint32_t>>> server_state;
  std::optional<std::string> file_state;
};

class LicenseMethod {
 public:
  virtual ~LicenseMethod() = default;
  Entitlements getEntitlements();
  LicenseMethodType getLicenseMethodType();
  virtual void checkLicense();
  virtual LicenseState getLicenseState();

 protected:
  LicenseMethod(Entitlements entitlements,
                LicenseMethodType license_method_type);
  Entitlements _entitlements;
  LicenseMethodType _license_method_type;
};

}  // namespace thirdai::licensing
