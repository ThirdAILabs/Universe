#include "CheckLicense.h"
#include <dataset/src/DataSource.h>
#include <licensing/src/entitlements/Entitlements.h>
#include <licensing/src/entitlements/RestrictionTree.h>
#include <optional>
#include <stdexcept>
#include <unordered_set>

// This file is only linked when the feature flag THIRDAI_CHECK_LICENSE is False

namespace thirdai::licensing {

void checkLicense() {}

void warnOnLicenseCall(const std::string& method) {
  std::cerr << "WARNING: calling 'licensing." + method +
                   "' on package built without license checks enabled."
            << std::endl;
}

Entitlements entitlements() { return Entitlements({FULL_ACCESS_ENTITLEMENT}); }

// The following functions have NOLINT because their arguments are pass-by-value
// even though they are not used. The values are used in CheckLicenseEnabled.cc,
// which requires the method signature to be pass by value for these functions
// as well
void activate(std::string api_key) {  // NOLINT
  (void)api_key;
  warnOnLicenseCall("activate");
}

void startHeartbeat(std::string heartbeat_url,  // NOLINT
                    const std::optional<uint32_t> heartbeat_timeout) {
  (void)heartbeat_url;
  (void)heartbeat_timeout;
  warnOnLicenseCall("start_heartbeat");
}

void setLicensePath(std::string license_path, bool verbose) {  // NOLINT
  (void)license_path;
  (void)verbose;
  warnOnLicenseCall("set_path");
}

void deactivate() { warnOnLicenseCall("deactivate"); }

LicenseState getLicenseState() {
  warnOnLicenseCall("_get_license_state");
  return {};
}

void setLicenseState(const LicenseState& state) {
  warnOnLicenseCall("_set_license_state");
  (void)state;
}

}  // namespace thirdai::licensing