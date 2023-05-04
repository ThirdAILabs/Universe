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
  Entitlements getEntitlements() { return _entitlements; }

  LicenseMethodType getLicenseMethodType() { return _license_method_type; }

  virtual void checkLicense();

  virtual LicenseState getLicenseState();

  virtual ~LicenseMethod() = default;

 protected:
  LicenseMethod(Entitlements entitlements,
                LicenseMethodType license_method_type)
      : _entitlements(std::move(entitlements)),
        _license_method_type(license_method_type) {}

  Entitlements _entitlements;
  LicenseMethodType _license_method_type;
};

}  // namespace thirdai::licensing
